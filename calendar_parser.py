"""
Calendar availability parser with format detection and confidence scoring.

This module provides enhanced parsing of calendar screenshots and PDFs,
supporting both Week View and Agenda View formats with confidence-based
slot extraction.
"""

import base64
import io
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from PIL import Image, ImageEnhance, ImageFilter


class CalendarFormat(Enum):
    """Detected calendar format type."""
    WEEK_VIEW = "week_view"
    AGENDA_VIEW = "agenda_view"
    UNKNOWN = "unknown"


@dataclass
class ParsedSlot:
    """A parsed availability slot with confidence score."""
    date: str  # YYYY-MM-DD
    start: str  # HH:MM
    end: str  # HH:MM
    confidence: float  # 0.0 to 1.0
    inferred_tz: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        result = {
            "date": self.date,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
        }
        if self.inferred_tz:
            result["inferred_tz"] = self.inferred_tz
        return result


@dataclass
class ParseResult:
    """Result of parsing a calendar image."""
    slots: List[ParsedSlot]
    detected_format: CalendarFormat
    format_confidence: float
    preprocessing_applied: List[str]
    raw_response: Optional[str] = None
    error: Optional[str] = None

    def to_legacy_format(self) -> List[Dict[str, str]]:
        """Convert to legacy slot format for backward compatibility."""
        return [slot.to_dict() for slot in self.slots]


@dataclass
class ParserConfig:
    """Configuration for the calendar parser."""
    debug_mode: bool = False
    pdf_dpi: int = 300
    apply_preprocessing: bool = True
    sharpness_factor: float = 1.3
    contrast_factor: float = 1.1
    min_image_width: int = 1200
    business_hours_start: str = "08:00"
    business_hours_end: str = "18:00"
    min_slot_minutes: int = 30


# Format detection prompt
FORMAT_DETECTION_PROMPT = """Analyze this calendar image and determine its format.

CALENDAR FORMATS:
1. WEEK VIEW: A grid-based view where:
   - Days are displayed as COLUMNS (horizontally arranged)
   - Hours are displayed as ROWS (vertically on the left edge)
   - You can see multiple days side-by-side in a grid pattern
   - Events appear as colored blocks positioned within the grid

2. AGENDA VIEW: A list-based view where:
   - Events are listed vertically as text entries
   - Often shows date headers followed by event details
   - May show "Print to PDF" or "Weekly Agenda" style output
   - Events are described with text like "9:00 AM - Meeting with John"
   - Does NOT have a grid layout

Respond with ONLY valid JSON (no markdown):
{
  "format": "week_view" or "agenda_view" or "unknown",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}
"""

# Week View extraction prompt (enhanced)
WEEK_VIEW_PROMPT = """You are extracting FREE/AVAILABLE time slots from an Outlook-style WEEK VIEW calendar.

CALENDAR LAYOUT:
- This is a week view: days are shown as COLUMNS (Monday-Sunday left to right)
- Hours are shown as ROWS on the LEFT EDGE (e.g., 07, 08, 09... or 7:00, 8:00, 9:00...)
- The date range header shows which week is displayed (e.g., '26 - 30 January 2026')
- Each column has the day name AND date number at the top (e.g., 'Monday' with '26' below it)

HOW TO IDENTIFY FREE vs BUSY TIME:
- BUSY TIME: Colored blocks (blue, orange, purple, etc.) with meeting titles - person is NOT available
- FREE TIME: White/blank areas - person IS available
- FREE slots are the GAPS between busy blocks within business hours

EXTRACTION PROCESS:
1. Read the date range from the header to get the correct year and dates
2. For each weekday column (Mon-Fri), scan from top to bottom
3. Identify all BUSY blocks by their position and the hour markers on the left
4. FREE slots are the continuous white/blank regions BETWEEN busy blocks
5. Use the hour markers on the left edge to determine precise start/end times
6. If a meeting block spans from the 10:00 line to the 12:00 line, it covers 10:00-12:00, so the FREE slot before it ends at 10:00

EXAMPLE: If Monday Jan 26 shows meetings at 12:00-13:00 and 15:00-16:30, then free slots are:
- 08:00-12:00 (morning gap before first meeting)
- 13:00-15:00 (gap between meetings)
- 16:30-18:00 (afternoon gap after last meeting ends)

Return ONLY valid JSON (no markdown):
[
  {{"date": "YYYY-MM-DD", "start": "HH:MM", "end": "HH:MM", "confidence": 0.95, "inferred_tz": null}}
]

RULES:
1. Extract FREE slots (gaps between meetings) within {business_start}-{business_end}
2. Clamp slots to business hours (start at {business_start} if earlier, end at {business_end} if later)
3. EXCLUDE weekends (Saturday and Sunday) entirely
4. All-day "Out of Office", "OOO", "Unavailable" banner at top = NO free slots that day
5. Minimum slot duration: {min_slot_minutes} minutes
6. If a day has NO meetings, return one slot for the full business hours: {business_start}-{business_end}

{tz_instruction}

Return empty list [] if no free slots found."""

# Agenda View extraction prompt (new)
AGENDA_VIEW_PROMPT = """You are extracting FREE/AVAILABLE time slots from an Outlook-style AGENDA VIEW or printed calendar.

LAYOUT STRUCTURE:
- This is a LIST format, not a grid
- Dates appear as headers or section titles
- Events are listed as text entries under each date
- Event format is typically: "Time - Event Title" or "Time Range: Event"
- May show "No events" or be blank for free days

INTERPRETATION APPROACH:
1. Identify date headers/sections (e.g., "Monday, January 19, 2026" or "Mon 19")
2. For each date, list all BUSY events with their times
3. Calculate FREE gaps between events within business hours
4. Days with "No events" listed are fully available during business hours

CALCULATING FREE SLOTS:
- If first event starts at 10:00, free slot exists from {business_start} to 10:00
- If there's a gap between events (e.g., meeting ends 11:00, next starts 14:00), that's a free slot
- If last event ends at 16:00, free slot exists from 16:00 to {business_end}

CONFIDENCE SCORING:
- 0.95-1.0: Event times clearly stated, gaps calculated precisely
- 0.80-0.94: Event times readable but format slightly non-standard
- 0.60-0.79: Times partially visible or format unclear
- Below 0.60: Do not include - too uncertain

Return ONLY valid JSON (no markdown):
[
  {{"date": "YYYY-MM-DD", "start": "HH:MM", "end": "HH:MM", "confidence": 0.0-1.0, "inferred_tz": "timezone or null"}}
]

RULES:
1. Extract FREE slots (gaps between events) within {business_start}-{business_end}
2. Clamp slots to business hours
3. EXCLUDE weekends (Saturday and Sunday) entirely
4. All-day "Out of Office", "OOO", "Unavailable" = NO free slots that day
5. Minimum slot duration: {min_slot_minutes} minutes
6. "No events" or blank day = fully free during business hours

{tz_instruction}

Return empty list [] if no free slots found."""


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def strip_code_fences(content: str) -> str:
    """Remove markdown code fences from content."""
    content = content.strip()

    # Handle ```json or ``` at start
    if content.startswith("```"):
        # Find end of first line (the ```json line)
        first_newline = content.find("\n")
        if first_newline != -1:
            content = content[first_newline + 1:]
        else:
            # No newline, just strip the backticks
            content = content[3:]

    # Handle trailing ```
    if content.endswith("```"):
        content = content[:-3]

    return content.strip()


def preprocess_image(
    image: Image.Image,
    config: ParserConfig
) -> Tuple[Image.Image, List[str]]:
    """
    Apply preprocessing to improve OCR/vision accuracy.

    Returns:
        Tuple of (processed_image, list_of_applied_transformations)
    """
    if not config.apply_preprocessing:
        return image, []

    applied = []
    processed = image.copy()

    # Upscale if too small
    if processed.width < config.min_image_width:
        scale = config.min_image_width / processed.width
        new_width = int(processed.width * scale)
        new_height = int(processed.height * scale)
        processed = processed.resize((new_width, new_height), Image.Resampling.LANCZOS)
        applied.append(f"upscaled_{scale:.1f}x")

    # Apply sharpening
    if config.sharpness_factor != 1.0:
        enhancer = ImageEnhance.Sharpness(processed)
        processed = enhancer.enhance(config.sharpness_factor)
        applied.append(f"sharpened_{config.sharpness_factor}x")

    # Apply contrast adjustment
    if config.contrast_factor != 1.0:
        enhancer = ImageEnhance.Contrast(processed)
        processed = enhancer.enhance(config.contrast_factor)
        applied.append(f"contrast_{config.contrast_factor}x")

    return processed, applied


def pdf_to_images_enhanced(
    pdf_bytes: bytes,
    max_pages: int = 3,
    dpi: int = 300
) -> List[Image.Image]:
    """
    Convert PDF to images at higher DPI for better parsing accuracy.

    Args:
        pdf_bytes: Raw PDF file content
        max_pages: Maximum number of pages to convert
        dpi: DPI for rendering (default 300, up from 200)

    Returns:
        List of PIL Images
    """
    images: List[Image.Image] = []
    doc = None
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for i in range(min(len(doc), max_pages)):
            try:
                page = doc.load_page(i)
                pix = page.get_pixmap(dpi=dpi)
                img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                images.append(img)
            except Exception:
                # Skip problematic pages
                pass
    except Exception:
        pass
    finally:
        if doc:
            doc.close()
    return images


class CalendarParser:
    """
    Enhanced calendar parser with format detection and confidence scoring.

    Usage:
        parser = CalendarParser(openai_client, config)
        result = parser.parse_image(image, interviewer_tz, display_tz)
        slots = result.to_legacy_format()  # For backward compatibility
    """

    def __init__(self, openai_client: Any, config: Optional[ParserConfig] = None):
        """
        Initialize the parser.

        Args:
            openai_client: OpenAI client instance
            config: Parser configuration (uses defaults if None)
        """
        self.client = openai_client
        self.config = config or ParserConfig()
        self.model = "gpt-5.2"  # Default, can be overridden

    def set_model(self, model: str) -> None:
        """Set the OpenAI model to use."""
        self.model = model

    def detect_format(self, image: Image.Image) -> Tuple[CalendarFormat, float, str]:
        """
        Detect the calendar format from an image.

        Returns:
            Tuple of (format, confidence, reasoning)
        """
        if not self.client:
            return CalendarFormat.UNKNOWN, 0.0, "No OpenAI client"

        b64 = image_to_base64(image)

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": "You analyze calendar images and return JSON."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": FORMAT_DETECTION_PROMPT},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        ],
                    },
                ],
            )
            content = resp.choices[0].message.content.strip() if resp.choices else ""

            # Strip code fences if present
            content = strip_code_fences(content)

            data = json.loads(content) if content else {}

            format_str = data.get("format", "unknown").lower()
            confidence = float(data.get("confidence", 0.5))
            reasoning = data.get("reasoning", "")

            if format_str == "week_view":
                return CalendarFormat.WEEK_VIEW, confidence, reasoning
            elif format_str == "agenda_view":
                return CalendarFormat.AGENDA_VIEW, confidence, reasoning
            else:
                return CalendarFormat.UNKNOWN, confidence, reasoning

        except Exception as e:
            return CalendarFormat.UNKNOWN, 0.0, str(e)

    def _build_extraction_prompt(
        self,
        calendar_format: CalendarFormat,
        interviewer_tz: Optional[str],
        display_tz: Optional[str]
    ) -> str:
        """Build the appropriate extraction prompt based on detected format."""

        # Build timezone instruction
        tz_instruction = ""
        if interviewer_tz and display_tz and interviewer_tz != display_tz:
            tz_instruction = (
                f"\nTIMEZONE CONVERSION:\n"
                f"  Calendar shows times in: {interviewer_tz}\n"
                f"  Convert all times to: {display_tz}\n"
            )
        elif interviewer_tz:
            tz_instruction = f"\nTimes are in timezone: {interviewer_tz}. Return as shown.\n"

        # Select base prompt
        if calendar_format == CalendarFormat.AGENDA_VIEW:
            base_prompt = AGENDA_VIEW_PROMPT
        else:
            # Default to week view for unknown formats
            base_prompt = WEEK_VIEW_PROMPT

        # Fill in template variables
        return base_prompt.format(
            business_start=self.config.business_hours_start,
            business_end=self.config.business_hours_end,
            min_slot_minutes=self.config.min_slot_minutes,
            tz_instruction=tz_instruction
        )

    def _extract_slots(
        self,
        image: Image.Image,
        prompt: str
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Extract slots from image using the given prompt.

        Returns:
            Tuple of (slots_list, raw_response)
        """
        if not self.client:
            return [], ""

        b64 = image_to_base64(image)

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that returns strict JSON."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        ],
                    },
                ],
            )
            content = resp.choices[0].message.content.strip() if resp.choices else ""
            raw_response = content

            # Strip code fences if present
            content = strip_code_fences(content)

            slots = json.loads(content) if content else []
            return slots if isinstance(slots, list) else [], raw_response

        except json.JSONDecodeError:
            return [], content if 'content' in locals() else ""
        except Exception as e:
            return [], str(e)

    def _validate_and_filter_slots(
        self,
        raw_slots: List[Dict[str, Any]]
    ) -> List[ParsedSlot]:
        """
        Validate and filter slots, applying business rules.
        """
        valid_slots = []

        for s in raw_slots:
            if not isinstance(s, dict):
                continue
            if not all(k in s for k in ("date", "start", "end")):
                continue

            slot_date = str(s.get("date", ""))
            slot_start = str(s.get("start", ""))
            slot_end = str(s.get("end", ""))
            confidence = float(s.get("confidence", 0.8))  # Default confidence if not provided
            inferred_tz = s.get("inferred_tz")

            # Validate date format and exclude weekends
            try:
                dt = datetime.strptime(slot_date, "%Y-%m-%d")
                if dt.weekday() >= 5:  # Saturday=5, Sunday=6
                    continue
            except ValueError:
                continue

            # Clamp to business hours
            if slot_start < self.config.business_hours_start:
                slot_start = self.config.business_hours_start
            if slot_end > self.config.business_hours_end:
                slot_end = self.config.business_hours_end

            # Validate time range
            if slot_start >= slot_end:
                continue

            # Check minimum duration
            try:
                start_dt = datetime.strptime(slot_start, "%H:%M")
                end_dt = datetime.strptime(slot_end, "%H:%M")
                duration_minutes = (end_dt - start_dt).seconds / 60
                if duration_minutes < self.config.min_slot_minutes:
                    continue
            except ValueError:
                continue

            valid_slots.append(ParsedSlot(
                date=slot_date,
                start=slot_start,
                end=slot_end,
                confidence=confidence,
                inferred_tz=str(inferred_tz) if inferred_tz else None
            ))

        return valid_slots

    def parse_image(
        self,
        image: Image.Image,
        interviewer_timezone: Optional[str] = None,
        display_timezone: Optional[str] = None,
        skip_format_detection: bool = False,
        assumed_format: Optional[CalendarFormat] = None
    ) -> ParseResult:
        """
        Parse a calendar image to extract availability slots.

        Args:
            image: PIL Image of the calendar
            interviewer_timezone: Timezone shown in the calendar
            display_timezone: Timezone for output times
            skip_format_detection: Skip detection and use assumed_format
            assumed_format: Format to use if skipping detection

        Returns:
            ParseResult with slots, format info, and debug data
        """
        if not self.client:
            return ParseResult(
                slots=[],
                detected_format=CalendarFormat.UNKNOWN,
                format_confidence=0.0,
                preprocessing_applied=[],
                error="No OpenAI client available"
            )

        # Apply preprocessing
        processed_image, preprocessing_applied = preprocess_image(image, self.config)

        # Detect format (or use provided)
        if skip_format_detection and assumed_format:
            detected_format = assumed_format
            format_confidence = 1.0
            format_reasoning = "Format assumed by caller"
        else:
            detected_format, format_confidence, format_reasoning = self.detect_format(processed_image)

        # Build extraction prompt
        prompt = self._build_extraction_prompt(
            detected_format,
            interviewer_timezone,
            display_timezone
        )

        # Extract slots
        raw_slots, raw_response = self._extract_slots(processed_image, prompt)

        # Validate and filter
        valid_slots = self._validate_and_filter_slots(raw_slots)

        return ParseResult(
            slots=valid_slots,
            detected_format=detected_format,
            format_confidence=format_confidence,
            preprocessing_applied=preprocessing_applied,
            raw_response=raw_response if self.config.debug_mode else None
        )

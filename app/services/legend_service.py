import os
import tempfile
from typing import List, Tuple

from fastapi import Depends, HTTPException
from PIL import Image, ImageDraw
from sqlalchemy.orm import Session

from database import get_db
from models.legend import LegendItem, LegendTable
from models.page import PDFPage
from models.project import Project
from schemas.llm_schemas import (
    LegendBBoxLLMResponse,
    LegendBBoxVerificationResponse,
    LegendExtractionLLMResponse,
)
from services.llm_service import LLMService, get_llm_service
from services.storage_service import StorageService, get_storage_service

LEGEND_EXTRACTION_PROMPT = (
    "You are given a cropped legend table from an engineering drawing. "
    "Extract every row as JSON objects with the legend description and optional label "
    "if a text label exists. If there is no explicit label, set has_label to false "
    "and label_text to null. Do not hallucinate rows. Double-check accuracy."
)

LEGEND_BBOX_PROMPT = (
    "Analyze this engineering drawing and locate ALL legend tables/sections on this page. "
    "Legend tables typically contain lists of symbols, icons, or abbreviations with their descriptions. "
    "IMPORTANT: If there are MULTIPLE legend tables on this page (e.g., 'Electrical Symbols', 'Piping Legend', "
    "'General Notes'), you MUST identify ALL of them separately. "
    "For each legend table found, provide:\n"
    "- bbox_norm: Bounding box coordinates [ymin, xmin, ymax, xmax] normalized to 0-1000 scale\n"
    "- description: Brief label for this table (e.g., 'Electrical Legend', 'Valve Symbols')\n\n"
    "Return a list of ALL legend tables found. If only one exists, return a list with one item."
)

LEGEND_BBOX_VERIFICATION_PROMPT = (
    "Review this image where RED bounding boxes have been drawn around detected legend tables. "
    "Verify if ALL bounding boxes CORRECTLY and COMPLETELY capture ALL legend sections on this page. "
    "Legend tables typically contain symbols/icons with their descriptions. "
    "Check for:\n"
    "1. Are all legend tables on the page detected? (none missing)\n"
    "2. Does each box fully capture its respective table? (not too small/large)\n"
    "3. Are the boxes properly aligned? (not misplaced)\n\n"
    "If everything is correct, set is_correct=true. "
    "If ANY legend table is missing, incorrectly sized, or misaligned, set is_correct=false "
    "and provide the complete corrected list of ALL legend table bounding boxes."
)

LEGEND_BBOX_REFINEMENT_PROMPT_TEMPLATE = (
    "Previous attempt to locate legend tables was incorrect.\n"
    "Feedback from verification: {feedback}\n"
    "Previous attempts history: {history}\n\n"
    "Analyze this engineering drawing again and identify ALL legend tables on this page. "
    "Provide corrected bounding box coordinates [ymin, xmin, ymax, xmax] normalized to 0-1000 scale "
    "for EACH legend table found. Do not miss any legend sections."
)


class LegendService:
    """Handles legend detection and extraction via LLM."""

    def __init__(self, storage_service: StorageService, llm_service: LLMService):
        self.storage = storage_service
        self.llm = llm_service
    
    def _draw_bboxes_on_image(
        self, image_path: str, legend_tables: List, width: int, height: int
    ) -> str:
        """Draw red bounding boxes for all legend tables on image."""
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        
        for idx, table in enumerate(legend_tables):
            bbox_norm = table.bbox_norm if hasattr(table, 'bbox_norm') else table['bbox']
            ymin, xmin, ymax, xmax = bbox_norm
            y1 = int((ymin / 1000) * height)
            x1 = int((xmin / 1000) * width)
            y2 = int((ymax / 1000) * height)
            x2 = int((xmax / 1000) * width)
            
            # Draw rectangle with red outline
            draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
            
            # Optionally add label
            desc = getattr(table, 'description', None) or table.get('description')
            if desc:
                draw.text((x1 + 5, y1 + 5), f"{idx+1}. {desc}", fill="red")
        
        annotated_path = image_path.replace(".png", "_bbox.png")
        img.save(annotated_path)
        img.close()  # Close file handle to avoid Windows lock
        return annotated_path
    
    def detect_legend_bbox_iterative(
        self, page_image_path: str, page_width: int, page_height: int, max_iterations: int = 3
    ) -> Tuple[List[dict], List[dict]]:
        """
        Iteratively detect and verify ALL legend bounding boxes on the page.
        
        Returns:
            Tuple of (final_legend_tables_list, history_log)
            where each legend_tables item is {bbox_norm, description}
        """
        history = []
        
        for iteration in range(max_iterations):
            if iteration == 0:
                # Initial detection
                prompt = LEGEND_BBOX_PROMPT
                bbox_response: LegendBBoxLLMResponse = self.llm.detect_legend_bbox(
                    prompt, page_image_path
                )
                legend_tables = bbox_response.legend_tables
            else:
                # Refinement based on previous feedback
                history_str = "\n".join([
                    f"Attempt {i+1}: {len(h['tables'])} table(s), feedback={h['feedback']}"
                    for i, h in enumerate(history)
                ])
                prompt = LEGEND_BBOX_REFINEMENT_PROMPT_TEMPLATE.format(
                    feedback=history[-1]['feedback'],
                    history=history_str
                )
                bbox_response: LegendBBoxLLMResponse = self.llm.detect_legend_bbox(
                    prompt, page_image_path
                )
                legend_tables = bbox_response.legend_tables
            
            # Draw all bboxes on image
            annotated_image_path = self._draw_bboxes_on_image(
                page_image_path, legend_tables, page_width, page_height
            )
            
            # Verify with LLM
            verification: LegendBBoxVerificationResponse = self.llm.verify_legend_bbox(
                LEGEND_BBOX_VERIFICATION_PROMPT, annotated_image_path
            )
            
            history.append({
                "iteration": iteration + 1,
                "tables": [{"bbox": t.bbox_norm, "desc": t.description} for t in legend_tables],
                "is_correct": verification.is_correct,
                "feedback": verification.feedback
            })
            
            if verification.is_correct:
                print(f"✓ Legend bboxes verified correct on iteration {iteration + 1} ({len(legend_tables)} table(s) found)")
                return [{"bbox_norm": t.bbox_norm, "description": t.description} for t in legend_tables], history
            
            # Use suggested tables if provided
            if verification.suggested_legend_tables:
                legend_tables = verification.suggested_legend_tables
                history[-1]['tables'] = [{"bbox": t.bbox_norm, "desc": t.description} for t in legend_tables]
        
        # Max iterations reached, return last attempt
        print(f"⚠ Max iterations ({max_iterations}) reached without verification ({len(legend_tables)} table(s))")
        return [{"bbox_norm": t.bbox_norm, "description": t.description} for t in legend_tables], history

    def detect_legends(self, db: Session, project: Project) -> List[LegendTable]:
        pages = (
            db.query(PDFPage)
            .filter(PDFPage.project_id == project.id)
            .order_by(PDFPage.page_number)
            .all()
        )

        if not pages:
            raise HTTPException(status_code=400, detail="Project has no processed pages.")

        # Delete existing legend items first (to avoid foreign key constraint violation)
        existing_tables = db.query(LegendTable).filter(LegendTable.project_id == project.id).all()
        for table in existing_tables:
            db.query(LegendItem).filter(LegendItem.legend_table_id == table.id).delete()
        # Now delete the legend tables
        db.query(LegendTable).filter(LegendTable.project_id == project.id).delete()
        db.commit()

        legend_tables: List[LegendTable] = []
        for page in pages:
            with tempfile.TemporaryDirectory() as tmp_dir:
                local_image = os.path.join(tmp_dir, f"{page.id}.png")
                self.storage.download_file(page.image_url, local_image)

                try:
                    # Use iterative detection with self-verification for ALL legend tables on page
                    legend_tables_list, history = self.detect_legend_bbox_iterative(
                        local_image, page.width, page.height, max_iterations=3
                    )
                    print(f"{len(legend_tables_list)} legend table(s) detected on page {page.page_number} after {len(history)} iteration(s)")
                except Exception as e:
                    print(f"Legend detection failed for page {page.page_number}: {e}")
                    continue

                width = page.width or 0
                height = page.height or 0
                full_image = Image.open(local_image)

                # Process each detected legend table
                for table_idx, table_data in enumerate(legend_tables_list):
                    bbox_norm = table_data["bbox_norm"]
                    description = table_data.get("description", f"Legend {table_idx + 1}")
                    
                    if len(bbox_norm) != 4:
                        print(f"  Skipping invalid bbox for table {table_idx + 1}")
                        continue
                    
                    ymin, xmin, ymax, xmax = bbox_norm
                    y1 = int((ymin / 1000) * height)
                    x1 = int((xmin / 1000) * width)
                    y2 = int((ymax / 1000) * height)
                    x2 = int((xmax / 1000) * width)
                    bbox_abs = (x1, y1, x2, y2)

                    # Crop and save each legend table
                    cropped_path = os.path.join(tmp_dir, f"{page.id}_legend_{table_idx}.png")
                    cropped_image = full_image.crop((x1, y1, x2, y2))
                    cropped_image.save(cropped_path)
                    cropped_image.close()  # Close file handle to avoid Windows lock

                    upload_url = self.storage.upload_file(
                        local_path=cropped_path,
                        mime_type="image/png",
                        filename=f"{project.id}_{page.page_number}_legend_{table_idx}.png",
                        project_name=project.name,
                        project_id=str(project.id),
                    )

                    table = LegendTable(
                        project_id=project.id,
                        page_id=page.id,
                        bbox_normalized=bbox_norm,
                        bbox_absolute=bbox_abs,
                        cropped_image_url=upload_url,
                        extraction_status="detected",
                    )
                    db.add(table)
                    legend_tables.append(table)
                    print(f"  ✓ Saved legend table: {description}")
                
                # Close the full_image after processing all tables on this page
                full_image.close()

        db.commit()
        for table in legend_tables:
            db.refresh(table)
        return legend_tables

    def extract_legend_items(
        self, db: Session, legend_table: LegendTable
    ) -> List[LegendItem]:
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_image = os.path.join(tmp_dir, f"{legend_table.id}.png")
            self.storage.download_file(legend_table.cropped_image_url, local_image)

            llm_response: LegendExtractionLLMResponse = self.llm.extract_legends(
                LEGEND_EXTRACTION_PROMPT, local_image
            )

        db.query(LegendItem).filter(
            LegendItem.legend_table_id == legend_table.id
        ).delete()

        legend_items: List[LegendItem] = []
        for idx, item in enumerate(llm_response.legend_items, start=1):
            legend_item = LegendItem(
                legend_table_id=legend_table.id,
                description=item.description.strip(),
                label_text=item.label_text.strip() if item.label_text else None,
                order_index=idx,
                icon_bbox_status="pending",
            )
            db.add(legend_item)
            legend_items.append(legend_item)

        legend_table.extraction_status = "legends_extracted"
        db.add(legend_table)

        db.commit()
        for item in legend_items:
            db.refresh(item)
        db.refresh(legend_table)
        return legend_items


def get_legend_service(
    storage_service: StorageService = Depends(get_storage_service),
    llm_service: LLMService = Depends(get_llm_service),
) -> LegendService:
    return LegendService(storage_service=storage_service, llm_service=llm_service)

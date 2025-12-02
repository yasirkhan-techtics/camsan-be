from typing import List

from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session

from database import get_db
from models.detection import IconDetection, IconLabelMatch, LabelDetection
from models.page import PDFPage
from models.project import Project
from lib.icon_text_label_matcher import (
    Detection as MatcherDetection,
    assign_labels_to_icons_dynamic,
)
from services.tag_overlap_service import TagOverlapService, get_tag_overlap_service
from services.storage_service import StorageService, get_storage_service
from services.llm_service import LLMService, get_llm_service


class MatchingService:
    """Matches icons with labels based on detections."""

    def __init__(
        self,
        db: Session,
        tag_overlap_service: TagOverlapService = None,
    ):
        self.db = db
        self.tag_overlap_service = tag_overlap_service

    def match_icons_to_labels(
        self,
        project: Project,
        resolve_overlaps: bool = False,
    ) -> List[IconLabelMatch]:
        """
        Match icons to labels using distance-based matching.
        
        Note: Overlap resolution is now a separate phase and should be called
        before this method via the /resolve-tag-overlaps endpoint.
        
        Args:
            project: Project to process
            resolve_overlaps: Whether to automatically resolve tag overlaps (default: False)
            
        Returns:
            List of IconLabelMatch records
        """
        print(f"[MATCHING] Starting matching for project {project.id}")
        
        # Step 1: Resolve tag overlaps if explicitly requested (disabled by default)
        if resolve_overlaps and self.tag_overlap_service:
            print(f"[MATCHING] Resolving tag overlaps before matching...")
            overlap_result = self.tag_overlap_service.resolve_overlaps(project)
            print(f"[MATCHING] Overlap resolution: {overlap_result['tags_removed']} tags removed")
        
        pages = (
            self.db.query(PDFPage)
            .filter(PDFPage.project_id == project.id)
            .order_by(PDFPage.page_number)
            .all()
        )
        if not pages:
            raise HTTPException(status_code=400, detail="Project has no pages.")
        
        print(f"[MATCHING] Found {len(pages)} pages")

        match_ids = (
            self.db.query(IconLabelMatch.id)
            .join(IconDetection)
            .filter(IconDetection.project_id == project.id)
            .all()
        )
        if match_ids:
            match_id_values = [mid[0] for mid in match_ids]
            (
                self.db.query(IconLabelMatch)
                .filter(IconLabelMatch.id.in_(match_id_values))
                .delete(synchronize_session=False)
            )
            self.db.commit()
            print(f"[MATCHING] Deleted {len(match_id_values)} existing matches")

        created_matches: List[IconLabelMatch] = []
        matched_label_ids = set()

        for page in pages:
            print(f"[MATCHING] Processing page {page.page_number} (id={page.id})")
            
            # Only use verified icon detections
            icon_records = (
                self.db.query(IconDetection)
                .filter(
                    IconDetection.project_id == project.id,
                    IconDetection.page_id == page.id,
                    IconDetection.verification_status == "verified",
                )
                .all()
            )
            
            # Only use verified label detections
            label_records = (
                self.db.query(LabelDetection)
                .filter(
                    LabelDetection.project_id == project.id,
                    LabelDetection.page_id == page.id,
                    LabelDetection.verification_status == "verified",
                )
                .all()
            )
            
            print(f"[MATCHING] Page {page.page_number}: {len(icon_records)} verified icons, {len(label_records)} verified labels")

            if not icon_records:
                print(f"[MATCHING] Skipping page {page.page_number} - no verified icons")
                continue

            icon_tuple = [
                (
                    record,
                    MatcherDetection(
                        bbox=tuple(record.bbox),
                        confidence=record.confidence,
                        scale=record.scale,
                        rotation=record.rotation,
                        center=tuple(record.center),
                    ),
                )
                for record in icon_records
            ]
            
            # Debug: print icon centers
            for i, (rec, det) in enumerate(icon_tuple):
                print(f"[MATCHING]   Icon {i}: center={det.center}, bbox={det.bbox}")

            label_tuple = [
                (
                    record,
                    MatcherDetection(
                        bbox=tuple(record.bbox),
                        confidence=record.confidence,
                        scale=record.scale,
                        rotation=record.rotation,
                        center=tuple(record.center),
                        # Use tag_name from label_template, fallback to legend_item.label_text
                        text_label=(
                            record.label_template.tag_name 
                            if record.label_template and record.label_template.tag_name
                            else (
                                record.label_template.legend_item.label_text
                                if record.label_template and record.label_template.legend_item
                                else None
                            )
                        ),
                    ),
                )
                for record in label_records
            ]
            
            # Debug: print label centers
            for i, (rec, det) in enumerate(label_tuple):
                print(f"[MATCHING]   Label {i}: center={det.center}, bbox={det.bbox}, text_label={det.text_label}")

            matcher_labels = [lt[1] for lt in label_tuple]
            icon_detections = [it[1] for it in icon_tuple]
            if not matcher_labels:
                matcher_labels = []
            
            print(f"[MATCHING] Page dimensions: height={page.height}, width={page.width}")

            matched_labels, debug_info = assign_labels_to_icons_dynamic(
                icon_detections,
                matcher_labels,
                image_shape=(page.height or 1000, page.width or 1000),
            )
            
            print(f"[MATCHING] Matching result: learned_threshold={debug_info.get('learned_threshold')}, learned_ratio={debug_info.get('learned_ratio')}")
            print(f"[MATCHING] Assignments: {debug_info.get('assignments')}")
            print(f"[MATCHING] Matched labels: {matched_labels}")

            assignment_map = {
                (entry["icon_idx"], entry["text_idx"]): entry for entry in debug_info.get(
                    "assignments", []
                )
            }

            for idx, (icon_db, _) in enumerate(icon_tuple):
                match_info = [
                    key for key in assignment_map.keys() if key[0] == idx
                ]
                if match_info:
                    icon_idx, label_idx = match_info[0]
                    label_db = label_tuple[label_idx][0]
                    distance = assignment_map[(icon_idx, label_idx)]["distance"]
                    if distance is None:
                        distance = 0.0
                    else:
                        distance = float(distance)
                    confidence = matched_labels[idx] != "unknown"
                    match = IconLabelMatch(
                        icon_detection_id=icon_db.id,
                        label_detection_id=label_db.id,
                        distance=distance,
                        match_confidence=icon_db.confidence if confidence else 0.0,
                        match_method="distance",
                        match_status="matched",
                    )
                    matched_label_ids.add(label_db.id)
                else:
                    # Unmatched icon
                    match = IconLabelMatch(
                        icon_detection_id=icon_db.id,
                        label_detection_id=None,
                        distance=0.0,
                        match_confidence=0.0,
                        match_method="distance",
                        match_status="unmatched_icon",
                    )
                self.db.add(match)
                created_matches.append(match)

        self.db.commit()
        for match in created_matches:
            self.db.refresh(match)
        
        # Log summary
        matched_count = len([m for m in created_matches if m.match_status == "matched"])
        unmatched_count = len([m for m in created_matches if m.match_status == "unmatched_icon"])
        print(f"[MATCHING] Complete: {matched_count} matched, {unmatched_count} unmatched icons")
        
        return created_matches


def get_matching_service(
    db: Session = Depends(get_db),
    storage_service: StorageService = Depends(get_storage_service),
    llm_service: LLMService = Depends(get_llm_service),
) -> MatchingService:
    tag_overlap_service = TagOverlapService(
        db=db,
        storage_service=storage_service,
        llm_service=llm_service,
    )
    return MatchingService(db=db, tag_overlap_service=tag_overlap_service)


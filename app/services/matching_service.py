from typing import List, Optional, Dict
from uuid import UUID
from collections import defaultdict

from fastapi import Depends, HTTPException
from sqlalchemy import or_
from sqlalchemy.orm import Session, selectinload

from database import get_db
from models.detection import IconDetection, IconLabelMatch, IconTemplate, LabelDetection, LabelTemplate
from models.legend import LegendItem
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
        legend_item_ids: Optional[List[UUID]] = None,
    ) -> List[IconLabelMatch]:
        """
        Match icons to labels using distance-based matching.
        
        Note: Overlap resolution is now a separate phase and should be called
        before this method via the /resolve-tag-overlaps endpoint.
        
        Args:
            project: Project to process
            resolve_overlaps: Whether to automatically resolve tag overlaps (default: False)
            legend_item_ids: Optional list of legend item IDs to limit processing
            
        Returns:
            List of IconLabelMatch records
        """
        print(f"[MATCHING] Starting matching for project {project.id}")
        
        if legend_item_ids:
            print(f"[MATCHING] 📌 Filtering to {len(legend_item_ids)} selected legend item(s)")
        
        # Get template IDs for the selected legend items (if filtering)
        icon_template_ids = None
        label_template_ids = None
        if legend_item_ids:
            icon_template_ids = [
                t.id for t in self.db.query(IconTemplate.id).filter(
                    IconTemplate.legend_item_id.in_(legend_item_ids)
                ).all()
            ]
            label_template_ids = [
                t.id for t in self.db.query(LabelTemplate.id).filter(
                    LabelTemplate.legend_item_id.in_(legend_item_ids)
                ).all()
            ]
        
        # Step 1: Resolve tag overlaps if explicitly requested (disabled by default)
        if resolve_overlaps and self.tag_overlap_service:
            print(f"[MATCHING] Resolving tag overlaps before matching...")
            overlap_result = self.tag_overlap_service.resolve_overlaps(project, legend_item_ids=legend_item_ids)
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

        # Delete distance-based matches (Basic Matching results)
        # When legend_item_ids is specified, only delete matches for those items
        # Otherwise, delete all distance matches for the project
        if icon_template_ids or label_template_ids:
            # Filter deletion to selected legend items only
            # Build conditions for icon-based and label-based matches
            template_conditions = []
            if icon_template_ids:
                template_conditions.append(IconDetection.icon_template_id.in_(icon_template_ids))
            if label_template_ids:
                template_conditions.append(LabelDetection.label_template_id.in_(label_template_ids))
            
            match_ids = (
                self.db.query(IconLabelMatch.id)
                .outerjoin(IconDetection, IconLabelMatch.icon_detection_id == IconDetection.id)
                .outerjoin(LabelDetection, IconLabelMatch.label_detection_id == LabelDetection.id)
                .filter(
                    IconLabelMatch.match_method == "distance",
                    or_(*template_conditions)
                )
                .all()
            )
            filter_msg = f"for {len(legend_item_ids)} selected legend item(s)"
        else:
            # Delete ALL distance-based matches for the project
            # This includes matched, unmatched_icon, AND unassigned_tag records
            # Use outer joins to catch unassigned_tag matches (where icon_detection_id is NULL)
            match_ids = (
                self.db.query(IconLabelMatch.id)
                .outerjoin(IconDetection, IconLabelMatch.icon_detection_id == IconDetection.id)
                .outerjoin(LabelDetection, IconLabelMatch.label_detection_id == LabelDetection.id)
                .filter(
                    IconLabelMatch.match_method == "distance",
                    or_(
                        IconDetection.project_id == project.id,
                        LabelDetection.project_id == project.id,
                    )
                )
                .all()
            )
            filter_msg = "for entire project"
        
        if match_ids:
            match_id_values = [mid[0] for mid in match_ids]
            (
                self.db.query(IconLabelMatch)
                .filter(IconLabelMatch.id.in_(match_id_values))
                .delete(synchronize_session=False)
            )
            self.db.commit()
            print(f"[MATCHING] Deleted {len(match_id_values)} existing distance-based matches {filter_msg}")

        created_matches: List[IconLabelMatch] = []
        matched_label_ids = set()

        # Get all legend items for this project (or filter by legend_item_ids)
        legend_item_query = (
            self.db.query(LegendItem)
            .filter(LegendItem.project_id == project.id)
            .options(
                selectinload(LegendItem.icon_template),
                selectinload(LegendItem.label_templates),
            )
        )
        if legend_item_ids:
            legend_item_query = legend_item_query.filter(LegendItem.id.in_(legend_item_ids))
        legend_items = legend_item_query.all()
        
        print(f"[MATCHING] Processing {len(legend_items)} legend items")
        
        # Process each legend item separately to prevent cross-contamination
        for legend_item in legend_items:
            legend_item_name = legend_item.description or f"Item {legend_item.id}"
            icon_template = legend_item.icon_template
            label_templates = legend_item.label_templates or []
            
            if not icon_template:
                print(f"[MATCHING] Skipping legend item '{legend_item_name}' - no icon template")
                continue
            
            # Get label template IDs for this legend item
            item_label_template_ids = [lt.id for lt in label_templates]
            
            print(f"\n[MATCHING] === Processing Legend Item: '{legend_item_name}' ===")
            print(f"[MATCHING]   Icon template: {icon_template.id}")
            print(f"[MATCHING]   Label templates: {len(item_label_template_ids)}")
            
            # Process each page for this legend item
            for page in pages:
                # Get icon detections for THIS legend item on this page
                icon_records = (
                    self.db.query(IconDetection)
                    .filter(
                        IconDetection.project_id == project.id,
                        IconDetection.page_id == page.id,
                        IconDetection.verification_status == "verified",
                        IconDetection.icon_template_id == icon_template.id,
                    )
                    .all()
                )
                
                if not icon_records:
                    continue  # No icons for this legend item on this page
                
                # Get label detections for THIS legend item on this page
                if item_label_template_ids:
                    label_records = (
                        self.db.query(LabelDetection)
                        .filter(
                            LabelDetection.project_id == project.id,
                            LabelDetection.page_id == page.id,
                            LabelDetection.verification_status == "verified",
                            LabelDetection.label_template_id.in_(item_label_template_ids),
                        )
                        .options(
                            selectinload(LabelDetection.label_template).selectinload(
                                LabelTemplate.legend_item
                            )
                        )
                        .all()
                    )
                else:
                    label_records = []
                
                print(f"[MATCHING]   Page {page.page_number}: {len(icon_records)} icons, {len(label_records)} labels")

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

                matcher_labels = [lt[1] for lt in label_tuple]
                icon_detections = [it[1] for it in icon_tuple]
                if not matcher_labels:
                    matcher_labels = []

                matched_labels, debug_info = assign_labels_to_icons_dynamic(
                    icon_detections,
                    matcher_labels,
                    image_shape=(page.height or 1000, page.width or 1000),
                )
                
                print(f"[MATCHING]     Assignments: {len(debug_info.get('assignments', []))} matches")

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

        # Track unassigned tags (labels that weren't matched to any icon)
        # Query verified labels for this project (filtered by legend_item_ids if specified)
        label_query = (
            self.db.query(LabelDetection)
            .filter(
                LabelDetection.project_id == project.id,
                LabelDetection.verification_status == "verified",
            )
        )
        if label_template_ids:
            label_query = label_query.filter(LabelDetection.label_template_id.in_(label_template_ids))
        all_verified_labels = label_query.all()
        
        for label in all_verified_labels:
            if label.id not in matched_label_ids:
                # Create a match record for unassigned tag
                unassigned_match = IconLabelMatch(
                    icon_detection_id=None,  # No icon for this tag
                    label_detection_id=label.id,
                    distance=0.0,
                    match_confidence=0.0,
                    match_method="distance",
                    match_status="unassigned_tag",
                )
                self.db.add(unassigned_match)
                created_matches.append(unassigned_match)

        self.db.commit()
        for match in created_matches:
            self.db.refresh(match)
        
        # Log summary
        matched_count = len([m for m in created_matches if m.match_status == "matched"])
        unmatched_icon_count = len([m for m in created_matches if m.match_status == "unmatched_icon"])
        unassigned_tag_count = len([m for m in created_matches if m.match_status == "unassigned_tag"])
        print(f"[MATCHING] Complete: {matched_count} matched, {unmatched_icon_count} unmatched icons, {unassigned_tag_count} unassigned tags")
        
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


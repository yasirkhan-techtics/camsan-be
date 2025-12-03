"""
Cascade Reset Service

When re-running a processing stage, this service clears all downstream data
to ensure consistency. Each stage depends on the previous stage's output.

Pipeline stages:
0. Raw Detection - Creates IconDetection and LabelDetection
1. Overlap Removal - Sets verification_status='rejected' on overlapping labels
2. LLM Verification - Sets verification_status on all detections
3. Basic Matching - Creates IconLabelMatch records (match_method='distance')
4. Tag Matching (Phase 5) - Creates IconLabelMatch (match_method='llm_matched')
5. Icon Matching (Phase 6) - Creates new IconDetection + IconLabelMatch (match_method='llm_matched')
"""

from uuid import UUID
from sqlalchemy.orm import Session
from fastapi import Depends

from database import get_db
from models.detection import IconDetection, LabelDetection, IconLabelMatch


class CascadeResetService:
    def __init__(self, db: Session):
        self.db = db
    
    def reset_from_stage(self, project_id: UUID, stage: int, detection_type: str = "all") -> dict:
        """
        Reset all data from the given stage onwards.
        
        Args:
            project_id: The project ID
            stage: The stage being re-run (0-5)
            detection_type: For stage 0, specify "icons", "labels", or "all"
        
        Returns:
            Dict with counts of deleted/reset records
        """
        result = {
            "icon_detections_deleted": 0,
            "label_detections_deleted": 0,
            "matches_deleted": 0,
            "icon_verification_reset": 0,
            "label_verification_reset": 0,
        }
        
        if stage == 0:
            # Re-running Raw Detection: Delete specific detection type
            result = self._reset_from_raw_detection(project_id, detection_type)
        elif stage == 1:
            # Re-running Overlap Removal: Reset label status, delete matches
            result = self._reset_from_overlap_removal(project_id)
        elif stage == 2:
            # Re-running LLM Verification: Reset specific detection type verification
            result = self._reset_from_llm_verification(project_id, detection_type)
        elif stage == 3:
            # Re-running Basic Matching: Delete ALL matches
            result = self._reset_from_basic_matching(project_id)
        elif stage == 4:
            # Re-running Tag Matching: Delete LLM matches only
            result = self._reset_from_tag_matching(project_id)
        elif stage == 5:
            # Re-running Icon Matching: Delete LLM matches (Phase 6 data)
            result = self._reset_from_icon_matching(project_id)
        
        self.db.commit()
        # Clear session cache to ensure subsequent queries get fresh data
        self.db.expire_all()
        return result
    
    def _reset_from_raw_detection(self, project_id: UUID, detection_type: str = "all") -> dict:
        """
        Delete detections and matches for a fresh start.
        
        Args:
            project_id: The project ID
            detection_type: "icons", "labels", or "all"
        """
        matches_deleted = 0
        icon_deleted = 0
        label_deleted = 0
        
        # Delete matches first (foreign key constraints)
        if detection_type in ("icons", "all"):
            matches_deleted = self.db.query(IconLabelMatch).filter(
                IconLabelMatch.icon_detection_id.in_(
                    self.db.query(IconDetection.id).filter(IconDetection.project_id == project_id)
                )
            ).delete(synchronize_session=False)
        
        # Delete icon detections
        if detection_type in ("icons", "all"):
            icon_deleted = self.db.query(IconDetection).filter(
                IconDetection.project_id == project_id
            ).delete(synchronize_session=False)
        
        # Delete label detections
        if detection_type in ("labels", "all"):
            label_deleted = self.db.query(LabelDetection).filter(
                LabelDetection.project_id == project_id
            ).delete(synchronize_session=False)
        
        print(f"🔄 [CASCADE RESET] Stage 0 (Raw Detection - {detection_type}): Deleted {icon_deleted} icon detections, {label_deleted} label detections, {matches_deleted} matches")
        
        return {
            "icon_detections_deleted": icon_deleted,
            "label_detections_deleted": label_deleted,
            "matches_deleted": matches_deleted,
            "icon_verification_reset": 0,
            "label_verification_reset": 0,
        }
    
    def _reset_from_overlap_removal(self, project_id: UUID) -> dict:
        """Reset label verification status and delete all matches."""
        print(f"🔄 [CASCADE RESET] Stage 1 (Overlap Removal) starting for project: {project_id}")
        
        # Check current state before reset
        verified_labels = self.db.query(LabelDetection).filter(
            LabelDetection.project_id == project_id,
            LabelDetection.verification_status == "verified"
        ).count()
        verified_icons = self.db.query(IconDetection).filter(
            IconDetection.project_id == project_id,
            IconDetection.verification_status == "verified"
        ).count()
        print(f"   Before reset: {verified_labels} verified labels, {verified_icons} verified icons")
        
        # Delete all matches
        matches_deleted = self.db.query(IconLabelMatch).filter(
            IconLabelMatch.icon_detection_id.in_(
                self.db.query(IconDetection.id).filter(IconDetection.project_id == project_id)
            )
        ).delete(synchronize_session=False)
        
        # Reset label verification status to 'pending'
        label_reset = self.db.query(LabelDetection).filter(
            LabelDetection.project_id == project_id
        ).update({"verification_status": "pending"}, synchronize_session=False)
        
        # Also reset icon verification status
        icon_reset = self.db.query(IconDetection).filter(
            IconDetection.project_id == project_id
        ).update({"verification_status": "pending"}, synchronize_session=False)
        
        print(f"🔄 [CASCADE RESET] Stage 1 (Overlap Removal): Reset {label_reset} labels, {icon_reset} icons to pending, deleted {matches_deleted} matches")
        
        return {
            "icon_detections_deleted": 0,
            "label_detections_deleted": 0,
            "matches_deleted": matches_deleted,
            "icon_verification_reset": icon_reset,
            "label_verification_reset": label_reset,
        }
    
    def _reset_from_llm_verification(self, project_id: UUID, detection_type: str = "all") -> dict:
        """
        Reset verification statuses and delete matches.
        
        Args:
            project_id: The project ID
            detection_type: "icons", "labels", or "all"
        """
        matches_deleted = 0
        icon_reset = 0
        label_reset = 0
        
        # Delete matches only if resetting icons (since matches depend on icon detections)
        if detection_type in ("icons", "all"):
            matches_deleted = self.db.query(IconLabelMatch).filter(
                IconLabelMatch.icon_detection_id.in_(
                    self.db.query(IconDetection.id).filter(IconDetection.project_id == project_id)
                )
            ).delete(synchronize_session=False)
        
        # Reset icon verification to pending
        if detection_type in ("icons", "all"):
            icon_reset = self.db.query(IconDetection).filter(
                IconDetection.project_id == project_id
            ).update({"verification_status": "pending"}, synchronize_session=False)
        
        # Reset label verification to pending, but KEEP overlap-rejected ones as rejected
        if detection_type in ("labels", "all"):
            label_reset = self.db.query(LabelDetection).filter(
                LabelDetection.project_id == project_id,
                LabelDetection.verification_status != "rejected"  # Keep overlap-rejected as is
            ).update({"verification_status": "pending"}, synchronize_session=False)
        
        print(f"🔄 [CASCADE RESET] Stage 2 (LLM Verification - {detection_type}): Reset {icon_reset} icons, {label_reset} labels to pending, deleted {matches_deleted} matches")
        
        return {
            "icon_detections_deleted": 0,
            "label_detections_deleted": 0,
            "matches_deleted": matches_deleted,
            "icon_verification_reset": icon_reset,
            "label_verification_reset": label_reset,
        }
    
    def _reset_from_basic_matching(self, project_id: UUID) -> dict:
        """Delete all matches (both distance and LLM) since basic matching is the foundation."""
        # Delete ALL matches - basic matching is the foundation for later stages
        # When re-running basic matching, we need a clean slate
        matches_deleted = self.db.query(IconLabelMatch).filter(
            IconLabelMatch.icon_detection_id.in_(
                self.db.query(IconDetection.id).filter(IconDetection.project_id == project_id)
            )
        ).delete(synchronize_session=False)
        
        print(f"🔄 [CASCADE RESET] Stage 3 (Basic Matching): Deleted {matches_deleted} matches (all types)")
        
        return {
            "icon_detections_deleted": 0,
            "label_detections_deleted": 0,
            "matches_deleted": matches_deleted,
            "icon_verification_reset": 0,
            "label_verification_reset": 0,
        }
    
    def _reset_from_tag_matching(self, project_id: UUID) -> dict:
        """Delete Tag Matching (Phase 5) matches only - match_method='llm_tag_for_icon'."""
        matches_deleted = self.db.query(IconLabelMatch).filter(
            IconLabelMatch.icon_detection_id.in_(
                self.db.query(IconDetection.id).filter(IconDetection.project_id == project_id)
            ),
            IconLabelMatch.match_method == "llm_tag_for_icon"
        ).delete(synchronize_session=False)
        
        print(f"🔄 [CASCADE RESET] Stage 4 (Tag Matching): Deleted {matches_deleted} tag-for-icon matches")
        
        return {
            "icon_detections_deleted": 0,
            "label_detections_deleted": 0,
            "matches_deleted": matches_deleted,
            "icon_verification_reset": 0,
            "label_verification_reset": 0,
        }
    
    def _reset_from_icon_matching(self, project_id: UUID) -> dict:
        """
        Delete Phase 6 data (Icon Matching) - match_method='llm_icon_for_tag'.
        
        This only deletes matches created by Icon Matching, preserving
        Tag Matching (Phase 5) results.
        """
        matches_deleted = self.db.query(IconLabelMatch).filter(
            IconLabelMatch.icon_detection_id.in_(
                self.db.query(IconDetection.id).filter(IconDetection.project_id == project_id)
            ),
            IconLabelMatch.match_method == "llm_icon_for_tag"
        ).delete(synchronize_session=False)
        
        print(f"🔄 [CASCADE RESET] Stage 5 (Icon Matching): Deleted {matches_deleted} icon-for-tag matches")
        
        return {
            "icon_detections_deleted": 0,
            "label_detections_deleted": 0,
            "matches_deleted": matches_deleted,
            "icon_verification_reset": 0,
            "label_verification_reset": 0,
        }


def get_cascade_reset_service(db: Session = Depends(get_db)) -> CascadeResetService:
    """
    Factory function for CascadeResetService.
    Uses FastAPI's dependency injection to share the database session with other services.
    """
    return CascadeResetService(db)


def cascade_reset_from_stage(db: Session, project_id: UUID, stage: int, detection_type: str = "all") -> dict:
    """
    Helper function to perform cascade reset using an existing db session.
    Call this directly when you need to ensure the same session is used.
    
    Args:
        db: Database session
        project_id: The project ID
        stage: The stage being re-run (0-5)
        detection_type: For stage 0, specify "icons", "labels", or "all"
    """
    service = CascadeResetService(db)
    return service.reset_from_stage(project_id, stage, detection_type)


from sqlalchemy.orm import Session
from app.models.db_models import NewProject, NewFeature


def save_requirement_results(
    db: Session,
    project_name: str,
    client_name: str | None,
    contact_info: str | None,
    results: list[dict],
    user_id: int | None = None,
) -> dict:
    
    try:
        
        project = NewProject(
            project_name=project_name,
            client_name=client_name,
            contact_info=contact_info,
            user_id=user_id,
        )
        db.add(project)
        db.flush()   

        
        rows = []
        for r in results:
            matched = r.get("matched_feature") or {}
            matched_id = matched.get("feature_id")

            row = NewFeature(
                project_id=project.id,
                requirement_name=r.get("requirement_name", ""),
                requirement_description=r.get("requirement_description", ""),
                matching_existing_feature_id=matched_id,
                match_status=r.get("status"),
                modification_needed=r.get("modification_needed"),
                confidence_score=r.get("confidence"),
            )
            db.add(row)
            rows.append(row)

        db.flush()
        db.commit()

        return {
            "project_id": project.id,
            "saved_count": len(rows),
        }

    except Exception:
        db.rollback()
        raise
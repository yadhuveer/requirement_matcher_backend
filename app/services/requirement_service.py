from sqlalchemy.orm import Session
from app.models.db_models import NewProject, NewFeature, NewFeatureMatch


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

        saved = 0
        for r in results:
            # one row per requirement: just the overall verdict
            row = NewFeature(
                project_id=project.id,
                requirement_name=r.get("requirement_name", ""),
                requirement_description=r.get("requirement_description", ""),
                match_status=r.get("status"),
                modification_needed=r.get("modification_needed"),
                explanation=r.get("explanation"),
                confidence_score=r.get("confidence"),
            )
            db.add(row)
            db.flush()  # get row.id for the join rows

            # composite matches: 0..N existing features per requirement
            for mf in r.get("matched_features") or []:
                completed_id = mf.get("feature_id")
                if completed_id is None:
                    continue
                db.add(
                    NewFeatureMatch(
                        new_feature_id=row.id,
                        completed_feature_id=completed_id,
                    )
                )

            saved += 1

        db.commit()

        return {
            "project_id": project.id,
            "saved_count": saved,
        }

    except Exception:
        db.rollback()
        raise

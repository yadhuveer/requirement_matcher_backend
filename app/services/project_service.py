from sqlalchemy import insert
from sqlalchemy.orm import Session
from app.models.db_models import CompletedProject, CompletedFeature


def create_project_with_features(
        db: Session,
        project_name: str,
        client_name: str | None,
        contact_info: str | None,
        features: list[dict],
        created_by: int | None = None,
) -> dict:
    try:
        project = CompletedProject(
            project_name=project_name,
            client_name=client_name,
            contact_info=contact_info,
            created_by=created_by,
        )
        db.add(project)
        db.flush()  # assigns project.id

        result_features = []
        if features:
            rows_data = [
                {
                    "project_id": project.id,
                    "name": f.get("name", ""),
                    "description": f["description"],
                    "domain": f.get("domain", ""),
                    "tech_details": f.get("tech_details", ""),
                }
                for f in features
            ]

            # ONE bulk INSERT ... RETURNING (SQLAlchemy insertmanyvalues) instead of
            # one INSERT per row. Turns hundreds of round-trips to a distant Neon DB
            # into a few. Same return contract as before (id + fields per feature).
            inserted = db.execute(
                insert(CompletedFeature).returning(
                    CompletedFeature.id,
                    CompletedFeature.name,
                    CompletedFeature.description,
                    CompletedFeature.domain,
                    CompletedFeature.tech_details,
                ),
                rows_data,
            ).all()

            result_features = [
                {
                    "id": r.id,
                    "name": r.name,
                    "description": r.description,
                    "domain": r.domain,
                    "tech_details": r.tech_details,
                }
                for r in inserted
            ]

        db.commit()

        return {
            "project_id": project.id,
            "features": result_features,
        }
    except Exception:
        db.rollback()
        raise

from sqlalchemy.orm import Session
from app.models.db_models import CompletedProject, CompletedFeature


def create_project_with_features(
        db:Session,
        project_name: str,
        client_name:str | None,
        contact_info:str | None,
        features: list[dict],
        created_by: int | None= None,
) -> dict:
    try:
        project = CompletedProject(
        project_name=project_name,
        client_name=client_name,
        contact_info=contact_info,
        created_by=created_by)

        db.add(project)
        db.flush()

        feature_rows = []
        for f in features:
            row = CompletedFeature(project_id=project.id, name=f.get("name",""),description=f["description"],domain=f.get("domain",""),tech_details=f.get("tech_details",""))
            db.add(row)
            feature_rows.append(row)

        db.flush()

        db.commit()

        return {
            "project_id":project.id,
            "features":[{
                "id":row.id,
                "name":row.name,
                "description":row.description,
                "domain":row.domain,
                "tech_details":row.tech_details
            } for row in feature_rows]
        }
    except Exception:
        db.rollback()
        raise
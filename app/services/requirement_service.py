from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload
from app.models.db_models import (
    NewProject,
    NewFeature,
    NewFeatureMatch,
    CompletedFeature,
)


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


# --------------------------------------------------------------------------- #
# History reads (per-user)                                                    #
# --------------------------------------------------------------------------- #

_STATUSES = ("exact_match", "needs_modification", "handle_manually")


def list_user_analyses(db: Session, user_id: int) -> list[dict]:
    """All of a user's past analyses, newest first, each with per-outcome counts."""
    projects = (
        db.query(NewProject)
        .filter(NewProject.user_id == user_id)
        .order_by(NewProject.created_at.desc())
        .all()
    )
    if not projects:
        return []

    project_ids = [p.id for p in projects]

    # One grouped query for all counts (avoids N+1 across projects).
    rows = (
        db.query(NewFeature.project_id, NewFeature.match_status, func.count(NewFeature.id))
        .filter(NewFeature.project_id.in_(project_ids))
        .group_by(NewFeature.project_id, NewFeature.match_status)
        .all()
    )
    agg: dict[int, dict[str, int]] = {}
    for pid, status, count in rows:
        agg.setdefault(pid, {})[status] = count

    analyses = []
    for p in projects:
        raw = agg.get(p.id, {})
        counts = {s: raw.get(s, 0) for s in _STATUSES}
        analyses.append(
            {
                "project_id": p.id,
                "project_name": p.project_name,
                "client_name": p.client_name,
                "created_at": p.created_at.isoformat() if p.created_at else None,
                "requirement_count": sum(raw.values()),
                "counts": counts,
            }
        )
    return analyses


def get_analysis_results(db: Session, project_id: int, user_id: int) -> dict | None:
    """Reconstruct one analysis's full results in the SAME shape the analyze
    endpoint returns. Scoped to the owning user; None if not found / not theirs."""
    project = (
        db.query(NewProject)
        .filter(NewProject.id == project_id, NewProject.user_id == user_id)
        .first()
    )
    if project is None:
        return None

    # Eager-load matches -> completed feature so this is a couple of queries, not N+1.
    features = (
        db.query(NewFeature)
        .filter(NewFeature.project_id == project_id)
        .options(
            joinedload(NewFeature.matches).joinedload(NewFeatureMatch.completed_feature)
        )
        .order_by(NewFeature.id)
        .all()
    )

    results = []
    for f in features:
        matched = []
        for m in f.matches:
            cf: CompletedFeature | None = m.completed_feature
            if cf is None:
                continue
            matched.append(
                {
                    "feature_id": cf.id,
                    "name": cf.name,
                    "description": cf.description,
                    "domain": cf.domain,
                }
            )
        results.append(
            {
                "requirement_name": f.requirement_name,
                "requirement_description": f.requirement_description,
                "status": f.match_status,
                "matched_features": matched,
                "explanation": f.explanation,
                "modification_needed": f.modification_needed,
                "confidence": f.confidence_score,
            }
        )

    return {
        "project_id": project.id,
        "project_name": project.project_name,
        "client_name": project.client_name,
        "created_at": project.created_at.isoformat() if project.created_at else None,
        "results": results,
    }

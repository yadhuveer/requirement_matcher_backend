from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Float, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    completed_projects = relationship("CompletedProject", back_populates="creator")
    new_projects = relationship("NewProject", back_populates="creator")


class CompletedProject(Base):
    __tablename__ = "completed_projects"

    id = Column(Integer, primary_key=True, index=True)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    project_name = Column(String, nullable=False)
    client_name = Column(String, nullable=True)
    contact_info = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    creator = relationship("User", back_populates="completed_projects")
    features = relationship("CompletedFeature", back_populates="project")




class CompletedFeature(Base):
    __tablename__ = "completed_features"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("completed_projects.id"), nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    domain = Column(String, nullable=True)
    tech_details = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    project = relationship("CompletedProject", back_populates="features")

class NewProject(Base):
    __tablename__ = "new_projects"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    project_name = Column(String, nullable=False)
    client_name = Column(String, nullable=True)
    contact_info = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    creator = relationship("User", back_populates="new_projects")
    features = relationship("NewFeature", back_populates="project")



class NewFeature(Base):
    """One client requirement + its overall verdict. Matched existing features
    (0..N of them) live in the new_feature_matches join table."""
    __tablename__ = "new_features"
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("new_projects.id"), nullable=False)
    requirement_name = Column(String, nullable=True)        # the requirement's name
    requirement_description = Column(Text, nullable=False)  # was "feature"
    match_status = Column(String, nullable=True)            # exact_match | needs_modification | handle_manually
    modification_needed = Column(Text, nullable=True)       # what to change (needs_modification)
    explanation = Column(Text, nullable=True)               # why it's covered (exact_match)
    confidence_score = Column(Float, nullable=True)         # one overall score from coverage
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    project = relationship("NewProject", back_populates="features")
    matches = relationship(
        "NewFeatureMatch",
        back_populates="new_feature",
        cascade="all, delete-orphan",
    )


class NewFeatureMatch(Base):
    """Join table (requirements.md §5): ONE requirement -> MANY matched existing features."""
    __tablename__ = "new_feature_matches"

    id = Column(Integer, primary_key=True, index=True)
    new_feature_id = Column(Integer, ForeignKey("new_features.id"), nullable=False, index=True)
    completed_feature_id = Column(Integer, ForeignKey("completed_features.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    new_feature = relationship("NewFeature", back_populates="matches")
    completed_feature = relationship("CompletedFeature")
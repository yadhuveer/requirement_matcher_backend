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
    __tablename__ = "new_features"
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("new_projects.id"), nullable=False)
    requirement_name = Column(String, nullable=True)        # NEW - the requirement's name
    requirement_description = Column(Text, nullable=False)  # was "feature"
    matching_existing_feature_id = Column(Integer, ForeignKey("completed_features.id"), nullable=True)
    match_status = Column(String, nullable=True)            # already have
    modification_needed = Column(Text, nullable=True)       # NEW - what to change
    confidence_score = Column(Float, nullable=True)         # already have
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    project = relationship("NewProject", back_populates="features")
from datetime import datetime
import sys

# Get the db from app module
from app import db

class GestureSession(db.Model):
    """
    A model representing a user session with gesture recognition.
    Used to track user interactions with the system.
    """
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(64), unique=True, nullable=False)
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime, nullable=True)
    device_info = db.Column(db.String(256), nullable=True)
    
    gestures = db.relationship('GestureDetection', backref='session', lazy='dynamic')
    
    def __repr__(self):
        return f'<GestureSession {self.session_id}>'

class GestureDetection(db.Model):
    """
    A model representing a single gesture detection event.
    Stores the detected gesture and associated metadata.
    """
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('gesture_session.id'), nullable=False)
    gesture_name = db.Column(db.String(64), nullable=False)

class User(db.Model):
    """User model for authentication and profile management"""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    bio = db.Column(db.String(500))
    expertise_level = db.Column(db.String(20))
    preferred_gestures = db.Column(db.String(200))
    
    # Relationships
    gesture_sessions = db.relationship('GestureSession', backref='user', lazy=True)
    achievements = db.relationship('Achievement', backref='user', lazy=True)

class Achievement(db.Model):
    """User achievements and badges"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(50), nullable=False)
    description = db.Column(db.String(200))
    earned_at = db.Column(db.DateTime, default=datetime.utcnow)
    badge_type = db.Column(db.String(20))

    detection_time = db.Column(db.DateTime, default=datetime.utcnow)
    confidence_score = db.Column(db.Float, nullable=True)
    
    def __repr__(self):
        return f'<GestureDetection {self.gesture_name}>'

class GestureStats(db.Model):
    """
    A model for aggregating gesture detection statistics.
    Used for analytics and improving the recognition system.
    """
    id = db.Column(db.Integer, primary_key=True)
    gesture_name = db.Column(db.String(64), unique=True, nullable=False)
    detection_count = db.Column(db.Integer, default=0)
    last_detected = db.Column(db.DateTime, nullable=True)
    average_confidence = db.Column(db.Float, default=0.0)
    
    def __repr__(self):
        return f'<GestureStats {self.gesture_name}>'
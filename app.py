import os
import logging
import uuid
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, request, session, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from sqlalchemy.orm import DeclarativeBase
import gesture_recognition as gr
import speech_generator as sg

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define base model class
class Base(DeclarativeBase):
    pass

# Initialize SQLAlchemy with base model
db = SQLAlchemy(model_class=Base)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default-secret-key")

# Configure database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize database with app
db.init_app(app)

# Initialize gesture recognition system

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            flash('Welcome back!', 'success')
            return redirect(url_for('profile'))
            
        flash('Invalid email or password', 'error')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        expertise_level = request.form.get('expertise_level')
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'error')
            return redirect(url_for('signup'))
            
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password),
            expertise_level=expertise_level
        )
        db.session.add(user)
        db.session.commit()
        
        session['user_id'] = user.id
        flash('Account created successfully!', 'success')
        return redirect(url_for('profile'))
        
    return render_template('signup.html')

@app.route('/profile')
@login_required
def profile():
    user = User.query.get(session['user_id'])
    return render_template('profile.html', user=user)

@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    user = User.query.get(session['user_id'])
    user.bio = request.form.get('bio')
    user.expertise_level = request.form.get('expertise_level')
    db.session.commit()
    flash('Profile updated successfully!', 'success')
    return redirect(url_for('profile'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

gesture_recognizer = gr.GestureRecognizer()

# Initialize speech generator
speech_generator = sg.SpeechGenerator()

# Import models after db is defined
with app.app_context():
    import models
    db.create_all()

@app.before_request
def before_request():
    """Create or retrieve session before each request."""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())

@app.route('/')
def index():
    """Render the main page of the application."""
    return render_template('index.html')

@app.route('/about')
def about():
    """Render the about page with information about supported gestures."""
    gestures = gesture_recognizer.get_supported_gestures()
    return render_template('about.html', gestures=gestures)

@app.route('/stats')
def stats():
    """Render the statistics page with gesture usage data."""
    return render_template('stats.html')
    
@app.route('/learning-mode')
def learning_mode():
    """Render the learning mode page with sign language education features."""
    return render_template('learning_mode.html')
    
@app.route('/custom-gestures')
def custom_gestures():
    """Render the custom gestures management page."""
    return render_template('custom_gestures.html')
    
@app.route('/gesture-library')
def gesture_library():
    """Render the gesture library page with comprehensive gesture collections."""
    # Common everyday gestures
    common_gestures = [
        {
            "display_name": "Thumbs Up",
            "name": "thumbs_up",
            "description": "Hand closed with thumb pointing upward, indicating approval or agreement.",
            "icon": "fas fa-thumbs-up",
            "accuracy": 95
        },
        {
            "display_name": "Thumbs Down",
            "name": "thumbs_down",
            "description": "Hand closed with thumb pointing downward, indicating disapproval or disagreement.",
            "icon": "fas fa-thumbs-down",
            "accuracy": 92
        },
        {
            "display_name": "Open Palm",
            "name": "palm_open",
            "description": "Hand open with fingers spread apart, often used as a stop or greeting gesture.",
            "icon": "fas fa-hand-paper",
            "accuracy": 90
        },
        {
            "display_name": "Closed Fist",
            "name": "fist",
            "description": "Hand closed tightly with all fingers folded into the palm, showing determination.",
            "icon": "fas fa-fist-raised",
            "accuracy": 88
        },
        {
            "display_name": "Peace Sign",
            "name": "peace",
            "description": "Index and middle fingers extended in a V shape, symbolizing peace or victory.",
            "icon": "fas fa-hand-peace",
            "accuracy": 93
        },
        {
            "display_name": "Point",
            "name": "pointer",
            "description": "Index finger extended with other fingers closed, used for pointing or indicating.",
            "icon": "fas fa-hand-point-up",
            "accuracy": 91
        }
    ]
    
    # Mathematical symbols as gestures
    math_gestures = [
        {
            "display_name": "Plus Sign",
            "name": "plus_sign",
            "symbol": "+",
            "description": "Gesture representing addition or incrementing values.",
            "operation": "Addition",
            "accuracy": 85
        },
        {
            "display_name": "Minus Sign",
            "name": "minus_sign",
            "symbol": "−",
            "description": "Gesture representing subtraction or decreasing values.",
            "operation": "Subtraction",
            "accuracy": 84
        },
        {
            "display_name": "Multiply",
            "name": "multiply",
            "symbol": "×",
            "description": "Gesture representing multiplication or scaling values.",
            "operation": "Multiplication",
            "accuracy": 82
        },
        {
            "display_name": "Divide",
            "name": "divide",
            "symbol": "÷",
            "description": "Gesture representing division or partitioning values.",
            "operation": "Division",
            "accuracy": 80
        },
        {
            "display_name": "Equals",
            "name": "equals",
            "symbol": "=",
            "description": "Gesture representing equality or assignment in equations.",
            "operation": "Equality",
            "accuracy": 88
        },
        {
            "display_name": "Greater Than",
            "name": "greater_than",
            "symbol": ">",
            "description": "Gesture representing the comparison operator 'greater than'.",
            "operation": "Comparison",
            "accuracy": 83
        }
    ]
    
    # Sign language alphabet gestures
    sign_gestures = [
        {
            "display_name": "Sign Letter A",
            "name": "sign_a",
            "letter": "A",
            "description": "ASL hand sign representation of the letter A.",
            "language": "American Sign Language",
            "accuracy": 87
        },
        {
            "display_name": "Sign Letter B",
            "name": "sign_b",
            "letter": "B",
            "description": "ASL hand sign representation of the letter B.",
            "language": "American Sign Language",
            "accuracy": 86
        },
        {
            "display_name": "Sign Letter C",
            "name": "sign_c",
            "letter": "C",
            "description": "ASL hand sign representation of the letter C.",
            "language": "American Sign Language",
            "accuracy": 85
        },
        {
            "display_name": "Sign Hello",
            "name": "sign_hello",
            "letter": "👋",
            "description": "ASL hand sign for the greeting 'Hello'.",
            "language": "American Sign Language",
            "accuracy": 90
        },
        {
            "display_name": "Sign Thank You",
            "name": "sign_thank_you",
            "letter": "🙏",
            "description": "ASL hand sign for expressing gratitude or 'Thank You'.",
            "language": "American Sign Language",
            "accuracy": 88
        },
        {
            "display_name": "Sign Yes",
            "name": "sign_yes",
            "letter": "✓",
            "description": "ASL hand sign for affirmation or 'Yes'.",
            "language": "American Sign Language",
            "accuracy": 89
        }
    ]
    
    return render_template('gesture_library.html', 
                          common_gestures=common_gestures,
                          math_gestures=math_gestures,
                          sign_gestures=sign_gestures)

@app.route('/video_feed')
def video_feed():
    """
    This route is used for streaming video from server to client.
    Not used in the current implementation as we're using client-side webcam capture.
    """
    return jsonify({"message": "Video feed endpoint is not used in this implementation. Using client-side webcam capture instead."})

@app.route('/api/gestures')
def get_gestures():
    """Return a list of supported gestures."""
    gestures = gesture_recognizer.get_supported_gestures()
    return jsonify(gestures)

@app.route('/api/speech-files')
def get_speech_files():
    """Get speech audio files for all gestures."""
    gestures = gesture_recognizer.get_supported_gestures()
    
    # Generate speech files for all gestures
    audio_map = speech_generator.get_speech_for_all_gestures(gestures)
    
    return jsonify({
        "status": "success",
        "speech_files": audio_map
    })

@app.route('/api/custom-gestures', methods=['POST'])
def add_custom_gesture():
    """Add a new custom gesture."""
    data = request.json
    if not data or 'name' not in data or 'description' not in data:
        return jsonify({
            "status": "error",
            "message": "Missing required fields (name, description)"
        }), 400
    
    name = data['name']
    description = data['description']
    
    # Try to save the new custom gesture
    gesture_data = data.get('gesture_data')
    success, message = gesture_recognizer.save_custom_gesture(name, description, gesture_data)
    
    if success:
        return jsonify({
            "status": "success",
            "message": message
        })
    else:
        return jsonify({
            "status": "error",
            "message": message
        }), 400

@app.route('/api/custom-gestures/<gesture_name>', methods=['DELETE'])
def delete_custom_gesture(gesture_name):
    """Delete a custom gesture by name."""
    success, message = gesture_recognizer.delete_custom_gesture(gesture_name)
    
    if success:
        return jsonify({
            "status": "success",
            "message": message
        })
    else:
        return jsonify({
            "status": "error",
            "message": message
        }), 404

@app.route('/api/start-session', methods=['POST'])
def start_session():
    """Start a new gesture recognition session."""
    # Get session ID from Flask session or create new one
    session_id = session.get('session_id', str(uuid.uuid4()))
    
    # Get device info from request headers
    user_agent = request.headers.get('User-Agent', '')
    
    # Check if session already exists in the database
    with app.app_context():
        existing_session = models.GestureSession.query.filter_by(session_id=session_id).first()
        
        if not existing_session:
            # Create new session record
            new_session = models.GestureSession(
                session_id=session_id,
                start_time=datetime.utcnow(),
                device_info=user_agent[:256]  # Limit to 256 chars
            )
            db.session.add(new_session)
            db.session.commit()
            logger.debug(f"Created new session: {session_id}")
        else:
            logger.debug(f"Using existing session: {session_id}")
    
    return jsonify({
        "status": "success",
        "session_id": session_id
    })

@app.route('/api/record-gesture', methods=['POST'])
def record_gesture():
    """Record a detected gesture."""
    data = request.json
    if not data or 'gesture' not in data:
        return jsonify({
            "status": "error",
            "message": "Missing gesture data"
        }), 400
    
    gesture_name = data['gesture']
    confidence = data.get('confidence', 1.0)
    session_id = session.get('session_id')
    
    if not session_id:
        return jsonify({
            "status": "error",
            "message": "No active session"
        }), 400
    
    try:
        with app.app_context():
            # Get session by session_id
            session_record = models.GestureSession.query.filter_by(session_id=session_id).first()
            
            if not session_record:
                logger.error(f"Session not found: {session_id}")
                return jsonify({
                    "status": "error",
                    "message": "Session not found"
                }), 404
            
            # Create new gesture detection record
            gesture_detection = models.GestureDetection(
                session_id=session_record.id,
                gesture_name=gesture_name,
                detection_time=datetime.utcnow(),
                confidence_score=confidence
            )
            db.session.add(gesture_detection)
            
            # Update stats
            stats = models.GestureStats.query.filter_by(gesture_name=gesture_name).first()
            if not stats:
                stats = models.GestureStats(
                    gesture_name=gesture_name,
                    detection_count=1,
                    last_detected=datetime.utcnow(),
                    average_confidence=confidence
                )
                db.session.add(stats)
            else:
                # Update average confidence and count
                new_count = stats.detection_count + 1
                stats.average_confidence = ((stats.average_confidence * stats.detection_count) + confidence) / new_count
                stats.detection_count = new_count
                stats.last_detected = datetime.utcnow()
            
            db.session.commit()
            logger.debug(f"Recorded gesture: {gesture_name} for session {session_id}")
            
            return jsonify({
                "status": "success",
                "message": f"Recorded {gesture_name}"
            })
    
    except Exception as e:
        logger.error(f"Error recording gesture: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "Internal server error"
        }), 500

@app.route('/api/end-session', methods=['POST'])
def end_session():
    """End the current gesture session."""
    session_id = session.get('session_id')
    
    if not session_id:
        return jsonify({
            "status": "error",
            "message": "No active session"
        }), 400
    
    try:
        with app.app_context():
            session_record = models.GestureSession.query.filter_by(session_id=session_id).first()
            
            if session_record:
                session_record.end_time = datetime.utcnow()
                db.session.commit()
                logger.debug(f"Ended session: {session_id}")
                
                return jsonify({
                    "status": "success",
                    "message": "Session ended"
                })
            else:
                return jsonify({
                    "status": "error",
                    "message": "Session not found"
                }), 404
    
    except Exception as e:
        logger.error(f"Error ending session: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "Internal server error"
        }), 500

@app.route('/api/stats')
def get_stats():
    """Get gesture detection statistics."""
    try:
        with app.app_context():
            stats = models.GestureStats.query.all()
            stats_data = []
            
            for stat in stats:
                stats_data.append({
                    "gesture_name": stat.gesture_name,
                    "detection_count": stat.detection_count,
                    "last_detected": stat.last_detected.isoformat() if stat.last_detected else None,
                    "average_confidence": round(stat.average_confidence, 2)
                })
            
            return jsonify({
                "status": "success",
                "stats": stats_data
            })
    
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "Error retrieving statistics"
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

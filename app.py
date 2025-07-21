from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
import tensorflow as tf
import numpy as np
import cv2
import uuid
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Required for session management

# Load trained deepfake detection model
image_model = tf.keras.models.load_model("models\images_model.h5")

# Ensure upload folder existS
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Dummy user database (hashed passwords for security)
users = {
    "shweta@gmail.com": generate_password_hash("123"),
    "abc@gmail.com": generate_password_hash("456"),
}

# ✅ Preprocess image for deepfake detection
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ✅ Splash Screen Route
@app.route("/")
def splash():
    """Show splash screen first, then redirect to login"""
    return render_template("splash.html")

# ✅ Redirect from splash to login
@app.route("/login", methods=["GET", "POST"])
def login():
    """Login Page"""
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        if email in users and check_password_hash(users[email], password):
            session["user"] = email
            return redirect(url_for("index"))

        flash("Invalid email or password", "danger")
        return redirect(url_for("login"))  # Redirect back to login after flashing

    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    """Signup Page"""
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        if email in users:
            flash("User already exists!", "warning")
            return redirect(url_for("signup"))

        users[email] = generate_password_hash(password)
        flash("Signup successful! Please login.", "success")
        return redirect(url_for("login"))

    return render_template("signup.html")

@app.route("/logout")
def logout():
    """Logout User"""
    session.pop("user", None)
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))

@app.route("/index", methods=["GET", "POST"])
def index():
    """Main Page (Only accessible after login)"""
    if "user" not in session:
        return redirect(url_for("login"))

    result = None
    filename = None
    confidence = None

    if request.method == "POST":
        if "image" in request.files:
            file = request.files["image"]
            if file and file.filename.lower().endswith(("png", "jpg", "jpeg")):
                # Generate unique filename
                unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
                filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
                file.save(filepath)

                # Process Image
                img = preprocess_image(filepath)
                if img is None:
                    flash("Error processing image", "danger")
                    return redirect(url_for("index"))

                # Deepfake Detection
                prediction = image_model.predict(img)[0][0]
                confidence = round(float(prediction) * 100, 2)
                result = f"Fake ({confidence}%)" if prediction > 0.5 else f"Real ({100 - confidence}%)"

                return render_template(
                    "result.html",
                    result=result,
                    filename=unique_filename,
                    confidence=confidence,
                    prediction=prediction
                )

            flash("Only PNG, JPG, and JPEG files are allowed.", "warning")

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)

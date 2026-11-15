import streamlit as st
import joblib
import numpy as np
from pathlib import Path

st.set_page_config(page_title="ScamShield PK", page_icon="🛡️", layout="centered")

MODEL_PATH = Path("scamshield_model.pkl")
VECTORIZER_PATH = Path("scamshield_vectorizer.pkl")

@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        return None, None
    # Only load pickle files created by your own trusted Colab notebook.
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

model, vectorizer = load_artifacts()

st.title("🛡️ ScamShield PK")
st.caption("Check suspicious SMS messages before you click, reply, or share information.")
st.info("This is an educational screening tool, not proof that a message is safe or fraudulent.")

if model is None:
    st.error("Model files are missing. Put scamshield_model.pkl and scamshield_vectorizer.pkl beside app.py.")
    st.stop()

message = st.text_area(
    "Paste the SMS message",
    height=150,
    placeholder="Example: Your account is blocked. Share your OTP immediately..."
)

if st.button("Scan message", type="primary", use_container_width=True):
    if not message.strip():
        st.warning("Please paste a message first.")
    else:
        features = vectorizer.transform([message])
        prediction = model.predict(features)[0]
        classes = list(model.classes_)
        spam_index = classes.index("spam")
        spam_score = float(model.predict_proba(features)[0][spam_index])

        if prediction == "spam":
            st.error("⚠️ Potential spam / scam")
        else:
            st.success("No spam pattern detected")

        st.metric("Model spam score", f"{spam_score * 100:.1f}%")
        if spam_score >= 0.75:
            risk = "High"
        elif spam_score >= 0.40:
            risk = "Medium"
        else:
            risk = "Low"
        st.write(f"**Risk band:** {risk}")

        # Show strongest positive TF-IDF × logistic-regression coefficient contributions.
        try:
            names = vectorizer.get_feature_names_out()
            coefficients = model.coef_[0]
            values = features.toarray()[0]
            contributions = values * coefficients
            ranked = np.argsort(contributions)[::-1]
            reasons = [
                names[i] for i in ranked
                if contributions[i] > 0
            ][:5]
            if reasons:
                st.write("**Terms contributing to a spam prediction:**")
                st.write(", ".join(reasons))
            else:
                st.write("No strong spam-associated terms were identified in this message.")
        except Exception:
            st.write("A term-level explanation could not be generated for this model.")

        st.warning(
            "A low score does not guarantee safety. Never share an OTP, PIN, password, "
            "or banking details through a message link. Verify requests using the "
            "organisation's official app or phone number."
        )

st.divider()
st.subheader("How it works")
st.write(
    "ScamShield PK uses TF-IDF text features and a Logistic Regression classifier "
    "trained on the SMS Spam Collection plus synthetic Pakistan-style examples."
)
st.caption(
    "Limitations: this model is trained on a small, mostly English dataset and synthetic "
    "local examples. It may miss new scams, Urdu/Roman Urdu messages, and legitimate "
    "messages that resemble spam. The displayed score is not a calibrated probability "
    "of fraud."
)

"""
app.py — Streamlit frontend for Biometric Face Recognition System
Tabs: Register | Authenticate | Users | Access Logs
"""

import streamlit as st
import requests
import base64
import pandas as pd
from datetime import datetime
from PIL import Image
from io import BytesIO
import plotly.graph_objects as go

st.set_page_config(
    page_title="Biometric System | Kelvin Mwangi",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://localhost:8000"

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.granted  { background:#EAF3DE; border-left:4px solid #639922;
            padding:16px; border-radius:8px; color:#27500A; font-weight:bold; }
.denied   { background:#FCEBEB; border-left:4px solid #E24B4A;
            padding:16px; border-radius:8px; color:#501313; font-weight:bold; }
.info-box { background:#E6F1FB; border-left:4px solid #185FA5;
            padding:12px; border-radius:8px; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def image_to_base64(img) -> str:
    buffer = BytesIO()
    if isinstance(img, bytes):
        img = Image.open(BytesIO(img))
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()


def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200, r.json()
    except Exception:
        return False, {}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Biometric System")
    st.markdown("Face Recognition | FastAPI | SQLite")
    st.markdown("---")
    online, health = check_api()
    if online:
        st.success(f"API online — {health.get('registered_users', 0)} users registered")
    else:
        st.error("API offline — start backend first")
        st.code("cd backend\npython main.py", language="bash")
    st.markdown("---")
    st.markdown("**Stack**")
    st.markdown("InsightFace · ArcFace · FastAPI  \nSQLite · Streamlit · OpenCV")
    st.markdown("---")
    st.markdown("**Kelvin Mwangi**  \n[GitHub](https://github.com/kevoflat)")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("👁 Biometric Face Recognition System")
st.markdown("Student and Employee Identification · Real-time face matching · Access logging")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "📷 Register", "🔐 Authenticate", "👥 Users", "📋 Access Logs"
])


# ── TAB 1: REGISTER ───────────────────────────────────────────────────────────
with tab1:
    st.subheader("Register New User")
    st.markdown("Capture face via webcam or upload a photo to enroll a new student or employee.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Personal Details")
        name       = st.text_input("Full Name", placeholder="e.g. John Kamau")
        student_id = st.text_input("Student / Employee ID", placeholder="e.g. CS/001/2024")
        role       = st.selectbox("Role", ["Student", "Employee", "Staff", "Visitor"])
        department = st.text_input("Department / Course", placeholder="e.g. Computer Science")

    with col2:
        st.markdown("#### Capture Face")
        capture_method = st.radio("Input method", ["Webcam", "Upload photo"], horizontal=True)
        img_data = None

        if capture_method == "Webcam":
            webcam_img = st.camera_input("Look directly at the camera")
            if webcam_img:
                img_data = webcam_img.getvalue()
                st.image(webcam_img, caption="Captured", width=280)
        else:
            uploaded = st.file_uploader("Upload clear face photo", type=["jpg", "jpeg", "png"])
            if uploaded:
                img_data = uploaded.getvalue()
                st.image(uploaded, caption="Uploaded", width=280)

    st.markdown("---")

    if st.button("Register User", type="primary", use_container_width=True):
        if not name or not student_id:
            st.error("Name and Student/Employee ID are required.")
        elif not img_data:
            st.error("Please capture or upload a face image.")
        else:
            with st.spinner("Extracting face embedding and registering..."):
                try:
                    r = requests.post(f"{API_URL}/register", json={
                        "name":       name,
                        "student_id": student_id,
                        "role":       role.lower(),
                        "department": department,
                        "image_b64":  image_to_base64(img_data)
                    }, timeout=120)
                    if r.status_code == 200:
                        st.success(f"✅ {r.json()['message']}")
                        st.balloons()
                    else:
                        st.error(f"Registration failed: {r.json().get('detail', 'Unknown error')}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API. Make sure the backend is running.")
                except requests.exceptions.ReadTimeout:
                    st.error("Timed out — server is still processing. Try again in a moment.")


# ── TAB 2: AUTHENTICATE ───────────────────────────────────────────────────────
with tab2:
    st.subheader("Authenticate User")
    st.markdown("Capture face to verify identity and grant or deny access.")

    col_a, col_b = st.columns([1, 1])

    with col_a:
        auth_method = st.radio("Input method", ["Webcam", "Upload photo"],
                               horizontal=True, key="auth_method")
        auth_img = None

        if auth_method == "Webcam":
            webcam_auth = st.camera_input("Look at the camera to authenticate", key="auth_cam")
            if webcam_auth:
                auth_img = webcam_auth.getvalue()
        else:
            auth_upload = st.file_uploader("Upload photo to verify",
                                           type=["jpg", "jpeg", "png"], key="auth_upload")
            if auth_upload:
                auth_img = auth_upload.getvalue()

        if st.button("Verify Identity", type="primary", use_container_width=True):
            if not auth_img:
                st.error("Please capture or upload a face image.")
            else:
                with st.spinner("Matching face against database..."):
                    try:
                        r = requests.post(f"{API_URL}/authenticate",
                                          json={"image_b64": image_to_base64(auth_img)},
                                          timeout=120)
                        st.session_state["auth_result"] = r.json()
                    except requests.exceptions.ConnectionError:
                        st.error("Cannot connect to API.")
                    except requests.exceptions.ReadTimeout:
                        st.error("Timed out — try again.")

    with col_b:
        st.markdown("#### Result")
        if "auth_result" in st.session_state:
            res = st.session_state["auth_result"]

            if res["status"] == "GRANTED":
                st.markdown(f"""
                <div class="granted">
                    ✅ ACCESS GRANTED<br><br>
                    <b>Name:</b> {res['name']}<br>
                    <b>ID:</b> {res['student_id']}<br>
                    <b>Role:</b> {res['role'].title()}<br>
                    <b>Department:</b> {res.get('department', 'N/A')}<br>
                    <b>Confidence:</b> {res['confidence']}%
                </div>
                """, unsafe_allow_html=True)
                st.success(res["message"])
            else:
                st.markdown(f"""
                <div class="denied">
                    ❌ ACCESS DENIED<br><br>
                    {res['message']}<br>
                    <b>Confidence:</b> {res.get('confidence', 0)}%
                </div>
                """, unsafe_allow_html=True)

            # Confidence gauge — threshold at 35 (ArcFace cosine × 100)
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=res.get("confidence", 0),
                title={"text": "Match Confidence (%)"},
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": "#639922" if res["status"] == "GRANTED" else "#E24B4A"},
                    "steps": [
                        {"range": [0,  35], "color": "#FCEBEB"},
                        {"range": [35, 60], "color": "#FAEEDA"},
                        {"range": [60, 100], "color": "#EAF3DE"},
                    ],
                    "threshold": {"line": {"color": "#333", "width": 2}, "value": 35}
                }
            ))
            fig.update_layout(height=240, margin=dict(t=30, b=10))
            st.plotly_chart(fig)
        else:
            st.markdown('<div class="info-box">Result will appear here after verification.</div>',
                        unsafe_allow_html=True)


# ── TAB 3: USERS ──────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Registered Users")
    col_r, col_d = st.columns([3, 1])

    with col_r:
        if st.button("🔄 Refresh", key="refresh_users"):
            st.rerun()
        try:
            r     = requests.get(f"{API_URL}/users", timeout=5)
            users = r.json()["users"]
            if users:
                df = pd.DataFrame(users).rename(columns={
                    "name": "Name", "student_id": "ID", "role": "Role",
                    "department": "Department", "registered_at": "Registered",
                    "last_seen": "Last Seen", "access_count": "Accesses"
                })
                df["Registered"] = pd.to_datetime(df["Registered"]).dt.strftime("%Y-%m-%d %H:%M")
                df["Last Seen"]  = pd.to_datetime(df["Last Seen"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
                st.metric("Total registered", len(df))
                st.dataframe(
                    df[["Name", "ID", "Role", "Department", "Registered", "Last Seen", "Accesses"]],
                    use_container_width=True, hide_index=True
                )
            else:
                st.info("No users registered yet.")
        except Exception:
            st.error("Could not fetch users. Is the API running?")

    with col_d:
        st.markdown("#### Delete User")
        del_id = st.text_input("Student/Employee ID to delete")
        if st.button("🗑️ Delete", type="secondary"):
            if del_id:
                try:
                    r = requests.delete(f"{API_URL}/delete",
                                        json={"student_id": del_id}, timeout=5)
                    if r.status_code == 200:
                        st.success(r.json()["message"])
                        st.rerun()
                    else:
                        st.error(r.json().get("detail"))
                except Exception:
                    st.error("API error")
            else:
                st.warning("Enter an ID to delete")


# ── TAB 4: ACCESS LOGS ────────────────────────────────────────────────────────
with tab4:
    st.subheader("Access Logs")
    st.markdown("Real-time log of all authentication attempts.")

    col_lim, col_btn = st.columns([4, 1])
    with col_lim:
        limit = st.slider("Show last N logs", 10, 200, 50)
    with col_btn:
        st.write("")
        st.write("")
        if st.button("🔄 Refresh"):
            st.rerun()

    try:
        r    = requests.get(f"{API_URL}/logs?limit={limit}", timeout=5)
        logs = r.json()["logs"]

        if logs:
            df_log = pd.DataFrame(logs)
            df_log["timestamp"]  = pd.to_datetime(df_log["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            df_log["confidence"] = df_log["confidence"].round(1).astype(str) + "%"

            def color_status(val):
                if val == "GRANTED": return "color: green; font-weight: bold"
                if val == "DENIED":  return "color: red;   font-weight: bold"
                return ""

            cols      = ["timestamp", "name", "student_id", "status", "confidence"]
            available = [c for c in cols if c in df_log.columns]

            st.dataframe(
                df_log[available].style.map(color_status, subset=["status"]),
                use_container_width=True, hide_index=True
            )

            # Stats row
            c1, c2, c3 = st.columns(3)
            granted = (df_log["status"] == "GRANTED").sum()
            denied  = (df_log["status"] == "DENIED").sum()
            c1.metric("Total attempts", len(df_log))
            c2.metric("✅ Granted", int(granted))
            c3.metric("❌ Denied",  int(denied))

            # CSV export
            st.divider()
            csv = df_log.to_csv(index=False)
            st.download_button(
                label="📥 Export Logs to CSV",
                data=csv,
                file_name=f"access_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No access logs yet. Authenticate a user to generate logs.")

    except Exception as e:
        st.error(f"Could not fetch logs. Is the API running? ({e})")
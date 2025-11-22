import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from plotly.subplots import make_subplots

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="AWIS - Workforce Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>

:root {
    --primary: #6366f1;
    --primary-light: #a5b4fc;
    --bg: #f8f9fc;
    --white: #ffffff;
    --text: #1f2937;
    --subtext: #6b7280;
    --card: #ffffff;
    --shadow: 0 4px 20px rgba(0,0,0,0.05);
    --radius: 14px;
    --transition: 0.25s ease;
}

/* Global Layout */
html, body {
    background-color: var(--bg) !important;
    font-family: 'Inter', sans-serif;
}

/* Remove Streamlit default padding */
section.main > div {
    padding-top: 1rem;
}

/* Headings */
h1, h2, h3 {
    font-weight: 700 !important;
    color: var(--text) !important;
}

h1 {
    font-size: 2.4rem !important;
    background: linear-gradient(135deg, var(--primary), #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Cards */
.block-container {
    padding: 2rem 3rem;
}

div[data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 700;
    color: var(--primary);
}

div[data-testid="stMetricLabel"] {
    font-size: 0.9rem !important;
    text-transform: uppercase;
    color: var(--subtext);
}

.css-1r6slb0, .stContainer {
    background: var(--white) !important;
    padding: 1.6rem !important;
    border-radius: var(--radius) !important;
    box-shadow: var(--shadow) !important;
    border: none !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #4f46e5, #7c3aed);
    padding: 2rem 1rem !important;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

/* Buttons */
.stButton > button {
    background: var(--primary);
    color: white;
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
    border: none;
    font-weight: 600;
    transition: var(--transition);
}

.stButton > button:hover {
    background: #4f46e5;
    transform: translateY(-2px);
}

/* Inputs */
input, select, textarea {
    border-radius: 10px !important;
    border: 1px solid #e5e7eb !important;
    padding: 0.6rem !important;
    transition: var(--transition);
}

input:focus, select:focus, textarea:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
}

/* Chat Messages */
.stChatMessage {
    background: var(--white) !important;
    border-radius: var(--radius) !important;
    padding: 1rem !important;
    box-shadow: var(--shadow);
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    background: white;
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background: var(--primary);
    color: white !important;
}

/* Hover animation for metrics */
.stMetric:hover {
    transform: scale(1.03);
    transition: var(--transition);
}

</style>
""", unsafe_allow_html=True)


# ==================== CONFIGURATION ====================
API_BASE_URL = "http://localhost:8000"



# Session state
if 'token' not in st.session_state:
    st.session_state.token = None
if 'user' not in st.session_state:
    st.session_state.user = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# ==================== HELPER FUNCTIONS ====================

def api_request(endpoint, method="GET", data=None, auth=False):
    """Make API request"""
    url = f"{API_BASE_URL}{endpoint}"
    headers = {}
    
    if auth and st.session_state.token:
        headers["Authorization"] = f"Bearer {st.session_state.token}"
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers, timeout=120)
        elif method == "POST":
            response = requests.post(url, json=data, headers=headers, timeout=120)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return None

def login(username, password):
    """Login to API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/auth/login",
            data={"username": username, "password": password},
            timeout=50
        )
        
        if response.status_code == 200:
            data = response.json()
            st.session_state.token = data["access_token"]
            st.session_state.user = data.get("user", {"username": username, "role": "user"})
            return True
        else:
            return False
    except Exception as e:
        st.error(f"Login failed: {e}")
        return False

def logout():
    """Logout"""
    st.session_state.token = None
    st.session_state.user = None
    st.session_state.chat_history = []
    st.rerun()

def create_metric_card(label, value, delta=None, icon="📊"):
    """Create a beautiful metric card"""
    st.markdown(f"""
    <div style="
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        border-left: 4px solid #667eea;
    ">
        <div style="color: #718096; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; margin-bottom: 0.5rem;">
            {icon} {label}
        </div>
        <div style="font-size: 2rem; font-weight: 700; color: #2d3748;">
            {value}
        </div>
        {f'<div style="color: #48bb78; font-size: 0.9rem; margin-top: 0.5rem;">▲ {delta}</div>' if delta else ''}
    </div>
    """, unsafe_allow_html=True)

# ==================== LOGIN PAGE ====================

if st.session_state.token is None:
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Logo/Title
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">🎯 AWIS</h1>
            <p style="color: #718096; font-size: 1.2rem;">Adaptive Workforce Intelligence System</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Login card
        st.markdown("""
        <div style="
            background: white;
            padding: 2rem;
            border-radius: 16px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        ">
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            st.markdown("### 🔐 Sign In")
            username = st.text_input("👤 Username", placeholder="Enter username")
            password = st.text_input("🔑 Password", type="password", placeholder="Enter password")
            
            col_a, col_b, col_c = st.columns([1, 2, 1])
            with col_b:
                submit = st.form_submit_button("Login", use_container_width=True)
            
            if submit:
                with st.spinner("Authenticating..."):
                    if login(username, password):
                        st.success("✅ Login successful!")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Demo credentials
        st.info("""
        **🎓 Demo Credentials:**
        
        **Admin Access:**
        - Username: `admin`
        - Password: `admin123`
        
        **HR Manager:**
        - Username: `hr_manager`
        - Password: `hr123`
        """)

else:
    # ==================== MAIN APP ====================
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0 2rem 0;">
            <h1 style="color: white !important; font-size: 2.5rem; margin: 0;">🎯 AWIS</h1>
            <p style="color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0;">Workforce Intelligence</p>
        </div>
        """, unsafe_allow_html=True)
        
        # User info card
        st.markdown(f"""
        <div style="
            background: rgba(255,255,255,0.1);
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            backdrop-filter: blur(10px);
        ">
            <div style="color: white; font-weight: 600; margin-bottom: 0.25rem;">👤 {st.session_state.user.get('username', 'User')}</div>
            <div style="color: rgba(255,255,255,0.7); font-size: 0.85rem;">🎭 {st.session_state.user.get('role', 'N/A').title()}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation
        st.markdown("<h3 style='color: white; margin-bottom: 1rem;'>📍 Navigation</h3>", unsafe_allow_html=True)
        
        page = st.radio(
            "Navigation",
            ["🏠 Dashboard", "👥 Attrition Analysis", "📈 Skill Forecast", "🔄 Career Mobility", "💬 AI Chat Assistant"],
            label_visibility="collapsed"
        )
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        if st.button("🚪 Logout", use_container_width=True):
            logout()
        
        # Footer
        st.markdown("""
        <div style="
            position: absolute;
            bottom: 1rem;
            left: 1rem;
            right: 1rem;
            text-align: center;
            color: rgba(255,255,255,0.6);
            font-size: 0.75rem;
        ">
            AWIS v1.0.0<br>
            © 2024 All Rights Reserved
        </div>
        """, unsafe_allow_html=True)
    
    # ==================== DASHBOARD PAGE ====================
    
    if page == "🏠 Dashboard":
        st.title("🏠 Executive Dashboard")
        st.markdown("Real-time workforce analytics and insights")
        
        # Fetch data
        with st.spinner("📊 Loading analytics..."):
            attrition_stats = api_request("/attrition/stats", auth=True)
            trending_skills = api_request("/forecast/trending?top_n=10", auth=True)
        
        if attrition_stats:
            # KPI Cards
            st.markdown("### 📊 Key Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "👥 Total Employees",
                    f"{attrition_stats['total_employees']:,}",
                    help="Total workforce size"
                )
            
            with col2:
                st.metric(
                    "🔴 High Risk",
                    attrition_stats['high_risk_count'],
                    f"{attrition_stats['high_risk_percentage']:.1f}%",
                    delta_color="inverse",
                    help="Employees at high risk of leaving"
                )
            
            with col3:
                st.metric(
                    "📈 Avg Risk",
                    f"{attrition_stats['average_attrition_probability']*100:.1f}%",
                    help="Average attrition probability"
                )
            
            with col4:
                st.metric(
                    "🟡 Medium Risk",
                    attrition_stats['medium_risk_count'],
                    help="Employees at moderate risk"
                )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Charts Row
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🎯 Risk Distribution")
                risk_dist = attrition_stats['risk_distribution']
                
                fig = go.Figure(data=[go.Pie(
                    labels=list(risk_dist.keys()),
                    values=list(risk_dist.values()),
                    hole=0.4,
                    marker=dict(colors=['#48bb78', '#f6ad55', '#fc8181']),
                    textinfo='label+percent',
                    textfont=dict(size=14, color='white'),
                    hovertemplate='<b>%{label}</b><br>Employees: %{value}<br>Percentage: %{percent}<extra></extra>'
                )])
                
                fig.update_layout(
                    showlegend=True,
                    height=400,
                    margin=dict(t=20, b=20, l=20, r=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Inter", size=12)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 🔥 Trending Skills")
                
                if trending_skills:
                    df = pd.DataFrame(trending_skills)
                    
                    # Color based on trend
                    colors = df['trend'].map({
                        'Rising': '#48bb78',
                        'Stable': '#667eea',
                        'Declining': '#fc8181'
                    })
                    
                    fig = go.Figure(data=[go.Bar(
                        x=df['growth_rate'],
                        y=df['skill_name'],
                        orientation='h',
                        marker=dict(
                            color=colors,
                            line=dict(color='white', width=2)
                        ),
                        text=df['growth_rate'].apply(lambda x: f"{x:+.1f}%"),
                        textposition='outside',
                        hovertemplate='<b>%{y}</b><br>Growth: %{x:.1f}%<extra></extra>'
                    )])
                    
                    fig.update_layout(
                        xaxis_title="Growth Rate (%)",
                        yaxis_title="",
                        height=400,
                        margin=dict(t=20, b=20, l=20, r=80),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Inter", size=12),
                        xaxis=dict(gridcolor='#e2e8f0'),
                        yaxis=dict(autorange="reversed")
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Additional insights
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 💡 Quick Insights")
            
            insight_col1, insight_col2, insight_col3 = st.columns(3)
            
            with insight_col1:
                st.info(f"""
                **🎯 Focus Area**
                
                {attrition_stats['high_risk_count']} employees need immediate attention with retention strategies.
                """)
            
            with insight_col2:
                if trending_skills:
                    rising_skills = [s for s in trending_skills if s['trend'] == 'Rising']
                    st.success(f"""
                    **📈 Growth Opportunities**
                    
                    {len(rising_skills)} skills are trending upward. Invest in training programs.
                    """)
            
            with insight_col3:
                st.warning(f"""
                **⚡ Action Required**
                
                Schedule 1-on-1 meetings with high-risk employees this week.
                """)
    
    # ==================== ATTRITION PAGE ====================
    
    elif page == "👥 Attrition Analysis":
        st.title("👥 Employee Attrition Analysis")
        st.markdown("Predict and prevent employee turnover with AI-powered insights")
        
        tab1, tab2 = st.tabs(["🔍 High Risk Employees", "🎯 Individual Prediction"])
        
        with tab1:
            st.markdown("### 🔍 High Risk Employee Dashboard")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                threshold = st.slider(
                    "🎚️ Risk Threshold",
                    0.0, 1.0, 0.7, 0.05,
                    help="Minimum probability to be considered high risk"
                )
            
            with col2:
                limit = st.number_input("📊 Show Top N", 5, 50, 20, 5)
            
            if st.button("🔄 Load High Risk Employees", use_container_width=True):
                with st.spinner("🔍 Analyzing employee data..."):
                    high_risk = api_request(
                        f"/attrition/high-risk?threshold={threshold}&limit={limit}",
                        auth=True
                    )
                
                if high_risk:
                    st.success(f"✅ Found {len(high_risk)} high-risk employees")
                    
                    for idx, emp in enumerate(high_risk, 1):
                        risk_color = {
                            'High': '#fc8181',
                            'Medium': '#f6ad55',
                            'Low': '#48bb78'
                        }.get(emp['risk_level'], '#cbd5e0')
                        
                        with st.expander(
                            f"#{idx} | {emp['employee_id']} - {emp['risk_level']} Risk ({emp['attrition_probability']*100:.1f}%)",
                            expanded=(idx <= 3)
                        ):
                            col1, col2 = st.columns([1, 1])
                            
                            with col1:
                                # Gauge chart
                                fig = go.Figure(go.Indicator(
                                    mode="gauge+number+delta",
                                    value=emp['attrition_probability'] * 100,
                                    domain={'x': [0, 1], 'y': [0, 1]},
                                    title={'text': "Attrition Risk %", 'font': {'size': 20}},
                                    delta={'reference': 50, 'increasing': {'color': "#fc8181"}},
                                    gauge={
                                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
                                        'bar': {'color': risk_color, 'thickness': 0.75},
                                        'bgcolor': "white",
                                        'borderwidth': 2,
                                        'bordercolor': "gray",
                                        'steps': [
                                            {'range': [0, 40], 'color': '#e6ffed'},
                                            {'range': [40, 70], 'color': '#fff3cd'},
                                            {'range': [70, 100], 'color': '#ffe6e6'}
                                        ],
                                        'threshold': {
                                            'line': {'color': "red", 'width': 4},
                                            'thickness': 0.75,
                                            'value': 70
                                        }
                                    }
                                ))
                                
                                fig.update_layout(
                                    height=300,
                                    margin=dict(l=20, r=20, t=50, b=20),
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    font={'family': "Inter", 'size': 14}
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                st.markdown("**💡 Retention Recommendations:**")
                                for rec in emp.get('recommendations', []):
                                    st.markdown(f"• {rec}")
        
        with tab2:
            st.markdown("### 🎯 Individual Employee Prediction")
            
            with st.form("prediction_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    employee_id = st.text_input("🆔 Employee ID", "EMP1001")
                    age = st.number_input("👤 Age", 20, 70, 35)
                    years_at_company = st.number_input("📅 Years at Company", 0, 40, 5)
                    monthly_income = st.number_input("💰 Monthly Income ($)", 20000, 300000, 50000, 1000)
                
                with col2:
                    performance_rating = st.select_slider(
                        "⭐ Performance Rating",
                        options=[1, 2, 3, 4, 5],
                        value=3
                    )
                    satisfaction_level = st.slider("😊 Satisfaction Level", 0.0, 1.0, 0.7, 0.1)
                    last_promotion_years = st.number_input("🏆 Years Since Last Promotion", 0, 20, 2)
                    training_hours = st.number_input("📚 Training Hours (Annual)", 0, 200, 40)
                
                submitted = st.form_submit_button("🔮 Predict Attrition Risk", use_container_width=True)
                
                if submitted:
                    data = {
                        "employee_id": employee_id,
                        "age": age,
                        "years_at_company": years_at_company,
                        "monthly_income": monthly_income,
                        "performance_rating": performance_rating,
                        "satisfaction_level": satisfaction_level,
                        "last_promotion_years": last_promotion_years,
                        "training_hours": training_hours
                    }
                    
                    with st.spinner("🔮 Analyzing..."):
                        result = api_request("/attrition/predict", method="POST", data=data, auth=True)
                    
                    if result:
                        risk_level = result['risk_level']
                        
                        if risk_level == 'High':
                            st.error(f"🔴 **{risk_level} Risk** - Immediate action required!")
                        elif risk_level == 'Medium':
                            st.warning(f"🟡 **{risk_level} Risk** - Monitor closely")
                        else:
                            st.success(f"🟢 **{risk_level} Risk** - Employee is stable")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Gauge
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=result['attrition_probability'] * 100,
                                title={'text': "Attrition Probability"},
                                gauge={'axis': {'range': [0, 100]},
                                       'bar': {'color': "#667eea"},
                                       'steps': [
                                           {'range': [0, 40], 'color': "#e6ffed"},
                                           {'range': [40, 70], 'color': "#fff3cd"},
                                           {'range': [70, 100], 'color': "#ffe6e6"}
                                       ]}
                            ))
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.markdown("**💡 Recommendations:**")
                            for rec in result.get('recommendations', []):
                                st.markdown(f"• {rec}")
    
    # ==================== FORECAST PAGE ====================
    
    elif page == "📈 Skill Forecast":
        st.title("📈 Skill Demand Forecasting")
        st.markdown("Predict future skill needs with AI-powered analytics")
        
        col1, col2, col3 = st.columns([3, 2, 1])
        
        with col1:
            skill_name = st.text_input(
                "🔍 Skill Name",
                "Python",
                help="Enter skill name to forecast (e.g., Python, Java, Leadership)"
            )
        
        with col2:
            months_ahead = st.select_slider(
                "📅 Forecast Period (Months)",
                options=[3, 6, 9, 12, 18, 24],
                value=6
            )
        
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            generate_btn = st.button("🚀 Generate", use_container_width=True)
        
        if generate_btn:
            data = {"skill_name": skill_name, "months_ahead": months_ahead}
            
            with st.spinner(f"🔮 Forecasting {skill_name} demand..."):
                result = api_request("/forecast/skill", method="POST", data=data, auth=True)
            
            if result:
                st.success(f"✅ Forecast generated for **{skill_name}**")
                
                # Metrics
                if result.get('trend'):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("📊 Trend", result['trend'])
                    
                    with col2:
                        if result.get('growth_rate') is not None:
                            st.metric("📈 Growth Rate", f"{result['growth_rate']:.1f}%")
                    
                    with col3:
                        forecast_df = pd.DataFrame(result['forecasts'])
                        avg_demand = forecast_df['forecasted_demand'].mean()
                        st.metric("💼 Avg Demand", f"{avg_demand:.0f}")
                
                # Chart
                forecast_df = pd.DataFrame(result['forecasts'])
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=forecast_df['date'],
                    y=forecast_df['forecasted_demand'],
                    mode='lines+markers',
                    name='Forecasted Demand',
                    line=dict(color='#667eea', width=3),
                    marker=dict(size=8, color='#764ba2'),
                    fill='tozeroy',
                    fillcolor='rgba(102, 126, 234, 0.1)',
                    hovertemplate='<b>Date:</b> %{x}<br><b>Demand:</b> %{y}<extra></extra>'
                ))
                
                fig.update_layout(
                    title=f"📈 {skill_name} Demand Forecast",
                    xaxis_title="Date",
                    yaxis_title="Demand (Positions)",
                    hovermode='x unified',
                    height=450,
                    margin=dict(l=20, r=20, t=60, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Inter", size=12),
                    xaxis=dict(gridcolor='#e2e8f0', showgrid=True),
                    yaxis=dict(gridcolor='#e2e8f0', showgrid=True)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Data table
                with st.expander("📋 View Detailed Forecast Data"):
                    st.dataframe(
                        forecast_df.style.background_gradient(cmap='Blues', subset=['forecasted_demand']),
                        use_container_width=True
                    )
    
    # ==================== MOBILITY PAGE ====================
    
    elif page == "🔄 Career Mobility":
        st.title("🔄 Career Mobility & Development")
        st.markdown("Discover career paths and skill development opportunities")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            employee_id = st.text_input(
                "🆔 Employee ID",
                "EMP1001",
                help="Enter employee ID to analyze career paths"
            )
        
        with col2:
            target_role = st.text_input(
                "🎯 Target Role (Optional)",
                "",
                help="Filter recommendations by role"
            )
        
        if st.button("🚀 Get Career Recommendations", use_container_width=True):
            data = {
                "employee_id": employee_id,
                "target_role": target_role if target_role else None
            }
            
            with st.spinner(f"🔍 Analyzing career paths for {employee_id}..."):
                result = api_request("/mobility/recommendations", method="POST", data=data, auth=True)
            
            if result:
                st.success(f"✅ Analysis complete for **{employee_id}**")
                
                # Current skills
                st.markdown("### 🎯 Current Skills")
                skills = result['current_skills']
                
                # Display skills in a nice grid
                skills_html = '<div style="display: flex; flex-wrap: wrap; gap: 8px;">'
                for skill in skills[:20]:
                    skills_html += f'''
                    <span style="
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 6px 12px;
                        border-radius: 20px;
                        font-size: 0.85rem;
                        font-weight: 600;
                    ">{skill}</span>
                    '''
                if len(skills) > 20:
                    skills_html += f'<span style="color: #718096;">+{len(skills)-20} more</span>'
                skills_html += '</div>'
                
                st.markdown(skills_html, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Similar employees
                if result['similar_employees']:
                    st.markdown("### 👥 Similar Employees")
                    
                    similar_col = st.columns(min(len(result['similar_employees']), 4))
                    
                    for idx, emp in enumerate(result['similar_employees'][:4]):
                        with similar_col[idx]:
                            st.markdown(f"""
                            <div style="
                                background: white;
                                padding: 1rem;
                                border-radius: 10px;
                                border: 2px solid #e2e8f0;
                                text-align: center;
                            ">
                                <div style="font-size: 2rem;">👤</div>
                                <div style="font-weight: 600; color: #2d3748;">{emp['employee_id']}</div>
                                <div style="font-size: 0.85rem; color: #718096; margin: 0.25rem 0;">{emp.get('job_role', 'N/A')}</div>
                                <div style="
                                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                    color: white;
                                    padding: 4px 8px;
                                    border-radius: 12px;
                                    font-size: 0.8rem;
                                    margin-top: 0.5rem;
                                    display: inline-block;
                                ">{emp['similarity_score']:.0%} Match</div>
                            </div>
                            """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Career recommendations
                st.markdown("### 🎯 Recommended Career Paths")
                
                for i, rec in enumerate(result['career_recommendations'][:5], 1):
                    with st.expander(
                        f"#{i} | {rec['target_role']} - {rec['skill_match_percentage']:.0f}% Match",
                        expanded=(i == 1)
                    ):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**🎯 Role Information**")
                            st.markdown(f"**Department:** {rec['department']}")
                            st.markdown(f"**Skill Match:** {rec['skill_match_percentage']:.1f}%")
                            st.markdown(f"**Skills to Learn:** {len(rec['missing_skills'])}")
                            
                            # Progress bar
                            st.progress(rec['skill_match_percentage'] / 100)
                        
                        with col2:
                            st.markdown("**📚 Skills to Develop**")
                            for skill in rec['missing_skills'][:8]:
                                st.markdown(f"• {skill}")
                            if len(rec['missing_skills']) > 8:
                                st.markdown(f"*...and {len(rec['missing_skills'])-8} more*")
    
    # ==================== AI CHAT PAGE ====================
    
    elif page == "💬 AI Chat Assistant":
        st.title("💬 AI Chat Assistant")
        st.markdown("Ask me anything about HR analytics, policies, or workforce insights")
        
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            # Display chat history
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("💬 Ask about HR policies, employee analytics, skill forecasting..."):
            # Add user message
            st.session_state.chat_history.append({
                "role": "user",
                "content": prompt
            })
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    data = {"message": prompt}
                    result = api_request("/chat/message", method="POST", data=data, auth=True)
                    
                    if result and result.get('success'):
                        response = result['response']
                        st.markdown(response)
                        
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response
                        })
                    else:
                        error_msg = "Sorry, I encountered an error. Please try again."
                        st.error(error_msg)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": error_msg
                        })
        
        # Sidebar for chat
        with st.sidebar:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown("### 💬 Chat Options")
            
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.chat_history = []
                api_request("/chat/reset", method="POST", auth=True)
                st.rerun()
            
            st.markdown("---")
            
            st.markdown("**💡 Try asking:**")
            example_questions = [
                "What is the remote work policy?",
                "Predict attrition for EMP1001",
                "Forecast Python demand",
                "Career paths for EMP1000",
                "Show trending skills"
            ]
            
            for question in example_questions:
                if st.button(f"📝 {question}", key=question, use_container_width=True):
                    # Add to chat
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": question
                    })
                    st.rerun()
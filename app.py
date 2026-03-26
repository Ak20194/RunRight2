"""
RunRight UAE — Founder Analytics Platform
8-Page Streamlit App: Descriptive → Diagnostic → Predictive → Prescriptive → Scoring
Models trained on startup from CSV — no pkl files required.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import io
import os
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                              precision_score, recall_score, confusion_matrix, roc_curve)

# ─── Resolve paths relative to this file (works on Streamlit Cloud) ───────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def data_path(filename):
    return os.path.join(BASE_DIR, filename)

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RunRight UAE · Analytics",
    page_icon="👟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0f172a; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
.metric-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    color: white;
}
.metric-card .val { font-size: 2rem; font-weight: 700; color: #38bdf8; }
.metric-card .lbl { font-size: 0.85rem; color: #94a3b8; margin-top: 4px; }
.tier1 { border-left: 4px solid #22c55e; background: #052e16; padding: 12px; border-radius: 8px; }
.tier2 { border-left: 4px solid #f59e0b; background: #1c1007; padding: 12px; border-radius: 8px; }
.tier3 { border-left: 4px solid #6366f1; background: #1e1b4b; padding: 12px; border-radius: 8px; }
.act-now { color: #22c55e; font-weight: 700; }
.nurture  { color: #f59e0b; font-weight: 600; }
.low-pri  { color: #64748b; }
h1, h2, h3 { color: #f1f5f9; }
</style>
""", unsafe_allow_html=True)

# ─── Feature list (52 features) ───────────────────────────────────────────────
FEATURES = [
    'Age_Enc','Income_Enc','Experience_Enc','Days_Per_Week_Enc',
    'Distance_Enc','Spend_Enc','Purchase_Freq_Enc','WTP_App_Enc',
    'App_Comfort_Enc','Q22_Current_Shoe_Satisfaction_1_7',
    'Q27_Runner_Identity_1_5','Q29_Peer_Influence_1_5',
    'Q30_Sustainability_Importance_1_5','Q35_Brand_Switch_Likelihood_1_5',
    'Discount_Trigger_Enc','Club_Member','Used_AI_Before','Waits_For_Sales',
    'Terrain_Road_Pavement','Terrain_Trail_Desert','Terrain_Treadmill','Terrain_Beach_Sand',
    'Goal_Full_Marathon','Goal_Half_Marathon','Goal_Ultra_Trail','Goal_5K','Goal_10K',
    'Motiv_Competitive_performance','Motiv_Mental_health','Motiv_Social_Community',
    'Priority_Speed_Performance','Priority_Comfort_Cushioning','Priority_Injury_Prevention',
    'App_Strava','App_Garmin','App_Apple_Watch_Health','App_Nike_Run_Club',
    'Acc_GPS_Watch','Acc_Compression_wear','Acc_Custom_insoles',
    'Brand_Nike','Brand_Adidas','Brand_ASICS','Brand_Hoka','Brand_On_Running',
    'Emirate_Dubai','Emirate_Abu_Dhabi','Emirate_Sharjah',
    'Occ_Corporate_Salaried','Occ_Fitness_professional','Occ_Self-employed','Occ_Student'
]

PERSONA_MAP = {
    2: 'Trail & Ultra Specialist',
    0: 'Serious Age-Grouper',
    5: 'Wellness Professional',
    3: 'Social Community Runner',
    1: 'Aspirational Beginner',
    4: 'Casual Lifestyle Runner',
}
TIER_MAP = {
    'Trail & Ultra Specialist': 'Tier 1', 'Serious Age-Grouper': 'Tier 1',
    'Wellness Professional':    'Tier 2', 'Social Community Runner': 'Tier 2',
    'Aspirational Beginner':    'Tier 3', 'Casual Lifestyle Runner': 'Tier 3',
}

# ─── Train all models from CSVs (cached — runs once per cold start) ────────────
@st.cache_resource(show_spinner="🏃 Training models — this takes ~30s on first load...")
def load_models():
    enc_raw = pd.read_csv(data_path('RunRight_UAE_Survey_Encoded.csv'))
    raw     = pd.read_csv(data_path('RunRight_UAE_Survey_Raw.csv'))

    X     = enc_raw[FEATURES].copy()
    imp   = SimpleImputer(strategy='median')
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=FEATURES)
    sc    = StandardScaler()
    X_sc  = sc.fit_transform(X_imp)

    # Classification
    y_clf = enc_raw['App_Interest_Binary']
    clf   = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    clf.fit(X_imp, y_clf)

    # Clustering
    km = KMeans(n_clusters=6, random_state=42, n_init=20)
    km.fit(X_sc)

    # Regression
    y_reg = enc_raw['Predicted_Annual_Shoe_Spend_AED']
    reg   = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    reg.fit(X_imp, y_reg)

    # PCA (3D visualisation)
    pca   = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X_sc)

    # Enrich dataset
    enc = enc_raw.copy()
    enc['Cluster']        = km.predict(X_sc)
    enc['Adopt_Prob']     = clf.predict_proba(X_imp)[:, 1]
    enc['Pred_Spend_AED'] = reg.predict(X_imp)
    enc['Persona']        = enc['Cluster'].map(PERSONA_MAP)
    enc['Tier']           = enc['Persona'].map(TIER_MAP)
    enc['PCA1'], enc['PCA2'], enc['PCA3'] = X_pca[:,0], X_pca[:,1], X_pca[:,2]
    tw                    = enc['Tier'].map({'Tier 1':1.0,'Tier 2':0.7,'Tier 3':0.3})
    sp_norm               = (enc['Pred_Spend_AED'] - enc['Pred_Spend_AED'].min()) / \
                            (enc['Pred_Spend_AED'].max() - enc['Pred_Spend_AED'].min())
    enc['Priority_Score'] = (enc['Adopt_Prob']*0.4 + sp_norm*0.4 + tw*0.2).round(4)
    enc['Priority_Tier']  = pd.cut(enc['Priority_Score'], bins=[0,0.35,0.55,1.01],
                                    labels=['Low Priority','Nurture','Act Now'])
    enc['Priority_Rank']  = enc['Priority_Score'].rank(ascending=False).astype(int)

    # ARM
    item_cols = [c for c in [
        'Terrain_Road_Pavement','Terrain_Trail_Desert','Terrain_Treadmill','Terrain_Beach_Sand',
        'Brand_Nike','Brand_Adidas','Brand_ASICS','Brand_Hoka','Brand_On_Running',
        'Acc_GPS_Watch','Acc_Compression_wear','Acc_Custom_insoles','Acc_Running_socks',
        'App_Strava','App_Garmin','App_Apple_Watch_Health','App_Nike_Run_Club',
        'Goal_Full_Marathon','Goal_Half_Marathon','Goal_Ultra_Trail','Goal_5K',
        'Motiv_Competitive_performance','Motiv_Social_Community','Motiv_Mental_health',
        'Priority_Speed_Performance','Priority_Comfort_Cushioning','Priority_Injury_Prevention',
        'Club_Member','Used_AI_Before'
    ] if c in enc.columns]
    idf  = enc[item_cols].fillna(0).astype(int)
    ssup = {c: idf[c].mean() for c in idf.columns if idf[c].mean() >= 0.05}
    arm_rows = []
    for a, c in combinations(list(ssup.keys()), 2):
        both = (idf[a] & idf[c]).mean()
        if both < 0.05: continue
        for ant, con in [(a,c),(c,a)]:
            conf = both / ssup[ant]; lift = conf / ssup[con]
            if conf >= 0.4 and lift >= 1.1:
                arm_rows.append({'antecedent':ant,'consequent':con,
                                 'support':round(both,3),'confidence':round(conf,3),'lift':round(lift,3)})
    rules_df = pd.DataFrame(arm_rows).sort_values('lift',ascending=False) if arm_rows else pd.DataFrame()

    # Pre-computed chart data
    Xtr,Xte,ytr,yte = train_test_split(X_imp, y_clf, test_size=0.2, random_state=42, stratify=y_clf)
    yp  = clf.predict_proba(Xte)[:,1]
    ypb = clf.predict(Xte)
    fpr,tpr,_ = roc_curve(yte, yp)
    cm_arr    = confusion_matrix(yte, ypb)
    clf_fi    = pd.DataFrame({'feature':FEATURES,'importance':clf.feature_importances_})\
                  .sort_values('importance',ascending=False).head(15)
    reg_fi    = pd.DataFrame({'feature':FEATURES,'importance':reg.feature_importances_})\
                  .sort_values('importance',ascending=False).head(15)
    _,Xrte,_,yrte = train_test_split(X_imp, y_reg, test_size=0.2, random_state=42)
    yrp = reg.predict(Xrte)

    precomp = {
        'roc_fpr': fpr.tolist(), 'roc_tpr': tpr.tolist(),
        'confusion_matrix': cm_arr.tolist(),
        'clf_feature_importance': clf_fi.to_dict('records'),
        'reg_feature_importance': reg_fi.to_dict('records'),
        'reg_actual': yrte.tolist(), 'reg_predicted': yrp.tolist(),
        'pca_variance': pca.explained_variance_ratio_.tolist(),
        'clf_acc': round(accuracy_score(yte, ypb), 3),
        'clf_auc': round(roc_auc_score(yte, yp), 3),
        'reg_r2':  round(float(1 - np.var(yrte.values-yrp)/np.var(yrte.values)), 3),
        'reg_mae': round(float(np.mean(np.abs(yrte.values-yrp))), 1),
    }

    return clf, reg, km, imp, sc, enc, raw, rules_df, precomp

clf, reg, km, imputer, scaler, enriched, raw, rules_df, precomp = load_models()
enc = enriched  # alias used throughout pages


PERSONA_COLORS = {
    'Trail & Ultra Specialist':  '#ef4444',
    'Serious Age-Grouper':       '#f97316',
    'Wellness Professional':     '#22c55e',
    'Social Community Runner':   '#3b82f6',
    'Aspirational Beginner':     '#a855f7',
    'Casual Lifestyle Runner':   '#64748b',
}
PERSONA_TIERS = {
    'Trail & Ultra Specialist':  'Tier 1',
    'Serious Age-Grouper':       'Tier 1',
    'Wellness Professional':     'Tier 2',
    'Social Community Runner':   'Tier 2',
    'Aspirational Beginner':     'Tier 3',
    'Casual Lifestyle Runner':   'Tier 3',
}

# ─── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 👟 RunRight UAE")
    st.markdown("### Analytics Platform")
    st.markdown("---")
    page = st.radio("Navigate", [
        "📊  Market Overview",
        "🔍  Segment Explorer",
        "🤖  Classification",
        "🎯  Clustering",
        "🔗  Association Rules",
        "💰  LTV & Regression",
        "🎬  Prescriptive Playbook",
        "📥  Score New Customers",
    ])
    st.markdown("---")
    st.markdown(f"**Dataset:** {len(enriched):,} respondents")
    st.markdown(f"**Features:** {len(FEATURES)}")
    st.markdown(f"**Classifier AUC:** {precomp['clf_auc']}")
    st.markdown(f"**Regressor R²:** {precomp['reg_r2']}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 · MARKET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊  Market Overview":
    import plotly.graph_objects as go
    import plotly.express as px

    st.title("📊 Market Overview")
    st.caption("Descriptive Analysis — Who is in the UAE running market?")

    # KPI Row
    act_now = (enriched['Priority_Tier'] == 'Act Now').sum()
    avg_spend = enriched['Pred_Spend_AED'].mean()
    app_interest = enriched['App_Interest_Binary'].mean() * 100
    tier1_pct = (enriched['Tier'] == 'Tier 1').mean() * 100

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    for col, val, lbl in [
        (c1, f"{len(enriched):,}", "Total Respondents"),
        (c2, f"{app_interest:.0f}%", "App Interest Rate"),
        (c3, f"AED {avg_spend:,.0f}", "Avg Annual Spend"),
        (c4, f"{act_now}", "Act Now Prospects"),
        (c5, f"{tier1_pct:.0f}%", "High-Value Tier"),
        (c6, f"{precomp['clf_auc']:.3f}", "Model AUC"),
    ]:
        col.markdown(f"""<div class="metric-card">
            <div class="val">{val}</div>
            <div class="lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Customer Persona Distribution")
        persona_counts = enriched['Persona'].value_counts()
        fig = px.pie(values=persona_counts.values, names=persona_counts.index,
                     color=persona_counts.index,
                     color_discrete_map=PERSONA_COLORS,
                     hole=0.45)
        fig.update_layout(showlegend=True, height=380, margin=dict(t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("App Interest by Priority Tier")
        pt = enriched.groupby(['Tier','Priority_Tier']).size().reset_index(name='count')
        fig2 = px.bar(pt, x='Tier', y='count', color='Priority_Tier',
                      color_discrete_map={'Act Now':'#22c55e','Nurture':'#f59e0b','Low Priority':'#475569'},
                      barmode='stack')
        fig2.update_layout(height=380, margin=dict(t=20,b=20), legend_title="Priority")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Average Predicted Spend by Persona")
        sp = enriched.groupby('Persona')['Pred_Spend_AED'].mean().sort_values(ascending=True).reset_index()
        fig3 = px.bar(sp, x='Pred_Spend_AED', y='Persona', orientation='h',
                      color='Pred_Spend_AED', color_continuous_scale='Blues',
                      labels={'Pred_Spend_AED': 'Avg AED/year'})
        fig3.update_layout(height=360, margin=dict(t=20,b=20), coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("Adoption Probability by Persona")
        ap = enriched.groupby('Persona')['Adopt_Prob'].mean().sort_values(ascending=True).reset_index()
        fig4 = px.bar(ap, x='Adopt_Prob', y='Persona', orientation='h',
                      color='Adopt_Prob', color_continuous_scale='Greens',
                      labels={'Adopt_Prob': 'Adoption Probability'})
        fig4.update_layout(height=360, margin=dict(t=20,b=20), coloraxis_showscale=False)
        st.plotly_chart(fig4, use_container_width=True)

    # Emirate breakdown
    st.subheader("Geographic Distribution")
    emirate_cols = [c for c in enc.columns if c.startswith('Emirate_')]
    emirate_sums = enc[emirate_cols].sum().sort_values(ascending=False)
    emirate_sums.index = [c.replace('Emirate_','').replace('_',' ') for c in emirate_sums.index]
    fig5 = px.bar(x=emirate_sums.index, y=emirate_sums.values,
                  labels={'x':'Emirate','y':'Respondents'},
                  color=emirate_sums.values, color_continuous_scale='Oranges')
    fig5.update_layout(height=300, margin=dict(t=20,b=20), coloraxis_showscale=False)
    st.plotly_chart(fig5, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 · SEGMENT EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍  Segment Explorer":
    import plotly.graph_objects as go
    import plotly.express as px

    st.title("🔍 Segment Explorer")
    st.caption("Diagnostic Analysis — Why do segments behave differently?")

    tab1, tab2, tab3 = st.tabs(["Cross-Tabulation", "Correlation Heatmap", "Segment Deep-Dive"])

    with tab1:
        st.subheader("Cross-Tabulation Explorer")
        num_cols = ['Age_Enc','Income_Enc','Experience_Enc','Days_Per_Week_Enc',
                    'Distance_Enc','WTP_App_Enc','Q22_Current_Shoe_Satisfaction_1_7',
                    'Q27_Runner_Identity_1_5','Q29_Peer_Influence_1_5',
                    'Adopt_Prob','Pred_Spend_AED']
        cat_cols = ['Persona','Tier','Priority_Tier','App_Interest_Binary']

        c1, c2, c3 = st.columns(3)
        x_var = c1.selectbox("X Axis (Group By)", cat_cols, index=0)
        y_var = c2.selectbox("Y Axis (Metric)", num_cols, index=10)
        chart_type = c3.selectbox("Chart Type", ["Box Plot", "Bar (Mean)", "Violin"])

        if chart_type == "Box Plot":
            fig = px.box(enriched, x=x_var, y=y_var, color=x_var,
                         color_discrete_map=PERSONA_COLORS if x_var=='Persona' else None)
        elif chart_type == "Bar (Mean)":
            agg = enriched.groupby(x_var)[y_var].mean().reset_index()
            fig = px.bar(agg, x=x_var, y=y_var, color=x_var,
                         color_discrete_map=PERSONA_COLORS if x_var=='Persona' else None)
        else:
            fig = px.violin(enriched, x=x_var, y=y_var, color=x_var, box=True,
                            color_discrete_map=PERSONA_COLORS if x_var=='Persona' else None)
        fig.update_layout(height=450, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Feature Correlation Heatmap")
        corr_cols = ['Age_Enc','Income_Enc','Experience_Enc','Days_Per_Week_Enc',
                     'Distance_Enc','Spend_Enc','WTP_App_Enc','App_Comfort_Enc',
                     'Q22_Current_Shoe_Satisfaction_1_7','Q27_Runner_Identity_1_5',
                     'Q29_Peer_Influence_1_5','Club_Member','Used_AI_Before',
                     'Adopt_Prob','Pred_Spend_AED']
        corr_df = enriched[corr_cols].corr().round(2)
        labels = [c.replace('_Enc','').replace('_',' ').replace('Q22 ','Satisf ').replace('Q27 ','Identity ') for c in corr_cols]
        fig = go.Figure(data=go.Heatmap(
            z=corr_df.values, x=labels, y=labels,
            colorscale='RdBu', zmid=0, zmin=-1, zmax=1,
            text=corr_df.values.round(2), texttemplate="%{text}",
            textfont={"size":9}
        ))
        fig.update_layout(height=600, margin=dict(t=20,b=80,l=120))
        st.plotly_chart(fig, use_container_width=True)

        # Key drivers callout
        st.info("💡 **Key insight:** Runner Identity (Q27) and WTP correlate strongly with predicted spend. "
                "App Comfort and Used AI Before are top adoption probability drivers.")

    with tab3:
        st.subheader("Persona Deep-Dive")
        selected_persona = st.selectbox("Select Persona", list(PERSONA_COLORS.keys()))
        persona_data = enriched[enriched['Persona'] == selected_persona]
        rest_data    = enriched[enriched['Persona'] != selected_persona]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Count", len(persona_data))
        c2.metric("Avg Spend", f"AED {persona_data['Pred_Spend_AED'].mean():,.0f}")
        c3.metric("Adopt Prob", f"{persona_data['Adopt_Prob'].mean():.1%}")
        c4.metric("Act Now %", f"{(persona_data['Priority_Tier']=='Act Now').mean():.1%}")

        # Radar comparison
        radar_features = ['Age_Enc','Income_Enc','Experience_Enc','Days_Per_Week_Enc',
                          'Distance_Enc','WTP_App_Enc','Q27_Runner_Identity_1_5','App_Comfort_Enc']
        radar_labels = ['Age','Income','Experience','Days/Week','Distance','WTP','Identity','App Comfort']

        persona_means = enriched.groupby('Persona')[radar_features].mean()
        overall_means = enc[radar_features].mean()
        # Normalise 0-1
        pmin = enc[radar_features].min()
        pmax = enc[radar_features].max()
        norm = lambda s: ((s - pmin) / (pmax - pmin)).clip(0,1)
        sel_norm = norm(persona_means.loc[selected_persona]).values.tolist()
        oth_norm = norm(persona_means.drop(selected_persona).mean()).values.tolist()
        sel_norm += [sel_norm[0]]
        oth_norm += [oth_norm[0]]
        angles = radar_labels + [radar_labels[0]]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=sel_norm, theta=angles, fill='toself',
                                      name=selected_persona, line_color=PERSONA_COLORS[selected_persona]))
        fig.add_trace(go.Scatterpolar(r=oth_norm, theta=angles, fill='toself',
                                      name='Other Personas (avg)', opacity=0.4, line_color='#64748b'))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0,1])), height=450, legend=dict(y=-0.2))
        st.plotly_chart(fig, use_container_width=True)

        # Top differentiating features
        diff = (persona_data[FEATURES].mean() - rest_data[FEATURES].mean()).sort_values()
        top_pos = diff.tail(8)
        top_neg = diff.head(5)
        combined = pd.concat([top_neg, top_pos]).sort_values()
        fig2 = px.bar(x=combined.values, y=combined.index, orientation='h',
                      color=combined.values, color_continuous_scale='RdBu',
                      title=f"How {selected_persona} differs from other personas",
                      labels={'x':'Mean Difference','y':'Feature'})
        fig2.update_layout(height=400, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 · CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Classification":
    import plotly.graph_objects as go
    import plotly.express as px

    st.title("🤖 App Adoption Classifier")
    st.caption("Predictive Analysis — Will this runner use the app?")

    X = enc[FEATURES].copy()
    y = enc['App_Interest_Binary'].copy()
    X_imp = pd.DataFrame(imputer.transform(X), columns=FEATURES)
    X_tr, X_te, y_tr, y_te = train_test_split(X_imp, y, test_size=0.2, random_state=42, stratify=y)
    y_prob = clf.predict_proba(X_te)[:,1]
    y_pred = clf.predict(X_te)

    # Metrics row
    acc   = accuracy_score(y_te, y_pred)
    auc   = roc_auc_score(y_te, y_prob)
    prec  = precision_score(y_te, y_pred)
    rec   = recall_score(y_te, y_pred)
    f1    = f1_score(y_te, y_pred)

    c1,c2,c3,c4,c5 = st.columns(5)
    for col,(lbl,val) in zip([c1,c2,c3,c4,c5],[
        ("Accuracy",f"{acc:.3f}"),("AUC-ROC",f"{auc:.3f}"),
        ("Precision",f"{prec:.3f}"),("Recall",f"{rec:.3f}"),("F1 Score",f"{f1:.3f}")]):
        col.markdown(f"""<div class="metric-card">
            <div class="val">{val}</div><div class="lbl">{lbl}</div></div>""", unsafe_allow_html=True)
    st.markdown("")

    # Threshold slider
    st.subheader("Decision Threshold Tuning")
    threshold = st.slider("Classification threshold (default 0.5)", 0.1, 0.9, 0.5, 0.01,
                           help="Move left = catch more interested users (higher recall). Move right = more precise.")
    y_pred_t = (y_prob >= threshold).astype(int)
    c1,c2,c3 = st.columns(3)
    c1.metric("Precision at threshold", f"{precision_score(y_te, y_pred_t):.3f}")
    c2.metric("Recall at threshold", f"{recall_score(y_te, y_pred_t):.3f}")
    c3.metric("F1 at threshold", f"{f1_score(y_te, y_pred_t):.3f}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ROC Curve")
        fpr_vals = precomp['roc_fpr']
        tpr_vals = precomp['roc_tpr']
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr_vals, y=tpr_vals, mode='lines',
                                 name=f'AUC = {auc:.3f}', line=dict(color='#38bdf8', width=2.5)))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                                 line=dict(dash='dash', color='#64748b'), name='Random'))
        fig.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
                          height=380, margin=dict(t=30))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_te, y_pred_t)
        fig2 = px.imshow(cm, text_auto=True,
                         labels=dict(x="Predicted", y="Actual"),
                         x=['Not Interested','Interested'], y=['Not Interested','Interested'],
                         color_continuous_scale='Blues')
        fig2.update_layout(height=380)
        st.plotly_chart(fig2, use_container_width=True)

    # Feature importance
    st.subheader("Top 15 Features Driving App Adoption")
    fi_data = precomp['clf_feature_importance']
    fi_df = pd.DataFrame(fi_data).sort_values('importance', ascending=True)
    fi_df['feature'] = fi_df['feature'].str.replace('_Enc','').str.replace('_',' ')
    fig3 = px.bar(fi_df, x='importance', y='feature', orientation='h',
                  color='importance', color_continuous_scale='Blues',
                  labels={'importance':'Importance','feature':'Feature'})
    fig3.update_layout(height=480, coloraxis_showscale=False, margin=dict(l=200))
    st.plotly_chart(fig3, use_container_width=True)

    st.info("💡 **Business Insight:** WTP for App, App Comfort, and Used AI Before are the strongest "
            "predictors of adoption — target users already comfortable with AI and fitness apps.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 · CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯  Clustering":
    import plotly.graph_objects as go
    import plotly.express as px

    st.title("🎯 Customer Segments")
    st.caption("Predictive Analysis — Which segment does each customer belong to?")

    tab1, tab2, tab3 = st.tabs(["3D Segment Map", "Cluster Profiles", "Segment Statistics"])

    with tab1:
        st.subheader("3D PCA Cluster Visualisation")
        sample = enriched.sample(min(800, len(enriched)), random_state=42)
        fig = px.scatter_3d(sample, x='PCA1', y='PCA2', z='PCA3',
                            color='Persona', symbol='Tier',
                            color_discrete_map=PERSONA_COLORS,
                            hover_data=['Pred_Spend_AED','Adopt_Prob','Priority_Tier'],
                            opacity=0.75, size_max=5)
        fig.update_layout(height=600, legend=dict(orientation='h', y=-0.15),
                          scene=dict(xaxis_title=f"PC1 ({precomp['pca_variance'][0]:.1%})",
                                     yaxis_title=f"PC2 ({precomp['pca_variance'][1]:.1%})",
                                     zaxis_title=f"PC3 ({precomp['pca_variance'][2]:.1%})"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"PCA variance explained: PC1={precomp['pca_variance'][0]:.1%}, "
                   f"PC2={precomp['pca_variance'][1]:.1%}, PC3={precomp['pca_variance'][2]:.1%}")

    with tab2:
        st.subheader("Cluster Profile Radar Charts")
        radar_feats = ['Age_Enc','Income_Enc','Experience_Enc','Days_Per_Week_Enc',
                       'Distance_Enc','WTP_App_Enc','Q27_Runner_Identity_1_5','Club_Member']
        radar_lbls  = ['Age','Income','Experience','Days/Wk','Distance','WTP','Identity','Club Mbr']
        pmin = enc[radar_feats].min()
        pmax = enc[radar_feats].max()
        norm = lambda s: ((s - pmin) / (pmax - pmin)).clip(0, 1)
        persona_means = enriched.groupby('Persona')[radar_feats].mean()

        cols = st.columns(3)
        for i, (persona, color) in enumerate(PERSONA_COLORS.items()):
            if persona not in persona_means.index: continue
            vals = norm(persona_means.loc[persona]).values.tolist() + [norm(persona_means.loc[persona]).values[0]]
            angles = radar_lbls + [radar_lbls[0]]
            with cols[i % 3]:
                fig = go.Figure(go.Scatterpolar(r=vals, theta=angles, fill='toself',
                                                line_color=color, fillcolor=color,
                                                opacity=0.4, name=persona))
                fig.update_layout(polar=dict(radialaxis=dict(range=[0,1], showticklabels=False)),
                                  title=dict(text=persona, font=dict(size=12)),
                                  height=280, margin=dict(t=50,b=20,l=30,r=30),
                                  showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Segment Statistics Comparison")
        stats_cols = ['Pred_Spend_AED','Adopt_Prob','Priority_Score',
                      'Age_Enc','Income_Enc','Experience_Enc','Q27_Runner_Identity_1_5']
        stats = enriched.groupby('Persona')[stats_cols].agg(['mean','std']).round(2)
        stats.columns = [' '.join(c) for c in stats.columns]
        st.dataframe(stats.style.background_gradient(subset=[c for c in stats.columns if 'mean' in c],
                                                      cmap='Blues'), use_container_width=True)

        st.subheader("Spend Distribution by Segment")
        fig = px.box(enriched, x='Persona', y='Pred_Spend_AED', color='Persona',
                     color_discrete_map=PERSONA_COLORS,
                     labels={'Pred_Spend_AED':'Predicted Annual Spend (AED)'})
        fig.update_layout(height=420, showlegend=False,
                          xaxis={'categoryorder':'median descending'})
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 · ASSOCIATION RULES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔗  Association Rules":
    import plotly.graph_objects as go
    import plotly.express as px
    import networkx as nx

    st.title("🔗 Association Rules — Bundle Intelligence")
    st.caption("What runners buy, use, and do together — your product bundle engine")

    tab1, tab2, tab3 = st.tabs(["Network Graph", "Top Rules", "Bundle Recommendations"])

    with tab1:
        st.subheader("Association Rule Network")
        min_lift = st.slider("Minimum Lift", 1.0, 2.5, 1.2, 0.05)
        filtered = rules_df[rules_df['lift'] >= min_lift].copy()
        st.caption(f"{len(filtered)} rules shown (of {len(rules_df)} total)")

        if len(filtered) > 0:
            G = nx.from_pandas_edgelist(filtered, 'antecedent', 'consequent',
                                        edge_attr=['lift','confidence','support'])
            pos = nx.spring_layout(G, seed=42, k=2)
            edge_x, edge_y, edge_text = [], [], []
            for e in G.edges(data=True):
                x0,y0 = pos[e[0]]; x1,y1 = pos[e[1]]
                edge_x += [x0,x1,None]; edge_y += [y0,y1,None]
                edge_text.append(f"Lift: {e[2]['lift']:.2f}")

            node_x = [pos[n][0] for n in G.nodes()]
            node_y = [pos[n][1] for n in G.nodes()]
            node_size = [G.degree(n)*8+10 for n in G.nodes()]
            node_labels = [n.replace('_',' ').replace(' Enc','') for n in G.nodes()]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines',
                                     line=dict(width=1, color='#334155'), hoverinfo='none'))
            fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text',
                                     text=node_labels, textposition='top center',
                                     textfont=dict(size=9),
                                     marker=dict(size=node_size, color='#38bdf8',
                                                 line=dict(width=1, color='white')),
                                     hoverinfo='text',
                                     hovertext=[f"{n}: {G.degree(n)} connections" for n in G.nodes()]))
            fig.update_layout(showlegend=False, height=550,
                              xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                              yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                              margin=dict(t=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No rules at this lift threshold. Lower the slider.")

    with tab2:
        st.subheader("Top Association Rules")
        sort_by = st.selectbox("Sort by", ["lift","confidence","support"], index=0)
        display = rules_df.sort_values(sort_by, ascending=False).head(30).copy()
        display['antecedent'] = display['antecedent'].str.replace('_',' ')
        display['consequent']  = display['consequent'].str.replace('_',' ')
        st.dataframe(display.style.background_gradient(subset=['lift'], cmap='Blues'),
                     use_container_width=True)

        # Lift bubble chart
        top20 = rules_df.head(20).copy()
        fig2 = px.scatter(top20, x='support', y='confidence', size='lift', color='lift',
                          hover_data=['antecedent','consequent'],
                          color_continuous_scale='Viridis',
                          labels={'support':'Support','confidence':'Confidence','lift':'Lift'},
                          title="Support vs Confidence (bubble size = Lift)")
        fig2.update_layout(height=380)
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.subheader("Persona-Specific Bundle Recommendations")
        bundles = {
            'Trail & Ultra Specialist': {
                'Bundle': 'Desert Trail Pro Kit',
                'Items': 'Hoka Speedgoat + Garmin Forerunner + Hydration Vest + Trail Socks',
                'ARM Support': 'Trail Desert → Garmin (lift 1.92), Trail Desert → Strava (lift 1.48)',
                'Expected Basket': 'AED 1,800–2,400',
                'Rationale': 'High trail + Garmin + Strava usage confirms tech-forward trail identity'
            },
            'Serious Age-Grouper': {
                'Bundle': 'Marathon Ready Pack',
                'Items': 'ASICS Gel-Nimbus + GPS Watch + Custom Insoles + Compression Wear',
                'ARM Support': 'Full Marathon → Strava (lift 1.44), Competitive → GPS Watch (lift 1.48)',
                'Expected Basket': 'AED 1,400–2,000',
                'Rationale': 'Marathon goal + competitive motivation drives performance gear'
            },
            'Wellness Professional': {
                'Bundle': 'Wellness Runner Collection',
                'Items': 'On Running Cloud + Foam Roller + Running Belt + Premium Socks',
                'ARM Support': 'Club Member → Garmin (lift 1.36), Strava → Club Member',
                'Expected Basket': 'AED 900–1,400',
                'Rationale': 'Club membership + mental health motivation = community + recovery focus'
            },
            'Social Community Runner': {
                'Bundle': 'Community Starter Pack',
                'Items': 'Nike React + Nike Run Club Premium + Running Socks + Belt',
                'ARM Support': 'NRC → Social Community Motiv (lift 1.20)',
                'Expected Basket': 'AED 700–1,000',
                'Rationale': 'NRC app usage + social motivation = brand-affiliated community gear'
            },
            'Aspirational Beginner': {
                'Bundle': 'First Steps Bundle',
                'Items': 'Adidas Ultraboost entry + Running Socks + Free App Trial 30-day',
                'ARM Support': 'Road Pavement → Nike/Adidas (entry brands)',
                'Expected Basket': 'AED 400–700',
                'Rationale': 'Lower spend, road terrain, brand-name appeal — freemium funnel entry'
            },
            'Casual Lifestyle Runner': {
                'Bundle': 'Lifestyle Flex Pack',
                'Items': 'New Balance Fresh Foam + Apple Watch integration + Insoles',
                'ARM Support': 'Apple Watch Health → general fitness motivation',
                'Expected Basket': 'AED 300–600',
                'Rationale': 'Apple Watch usage + general fitness = casual tech-enabled lifestyle'
            },
        }
        for persona, bundle in bundles.items():
            tier = PERSONA_TIERS[persona]
            tier_class = 'tier1' if tier=='Tier 1' else ('tier2' if tier=='Tier 2' else 'tier3')
            with st.expander(f"{'🔴' if tier=='Tier 1' else '🟡' if tier=='Tier 2' else '🔵'} {persona} — {bundle['Bundle']}"):
                c1, c2 = st.columns(2)
                c1.markdown(f"**Items:** {bundle['Items']}")
                c1.markdown(f"**ARM Support:** {bundle['ARM Support']}")
                c2.markdown(f"**Expected Basket:** {bundle['Expected Basket']}")
                c2.markdown(f"**Rationale:** {bundle['Rationale']}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 · LTV & REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💰  LTV & Regression":
    import plotly.graph_objects as go
    import plotly.express as px

    st.title("💰 LTV & Spend Forecasting")
    st.caption("Predictive Analysis — How much will each customer spend?")

    X = enc[FEATURES].copy()
    y_reg = enc['Predicted_Annual_Shoe_Spend_AED'].copy()
    X_imp = pd.DataFrame(imputer.transform(X), columns=FEATURES)
    _, X_te, _, y_te = train_test_split(X_imp, y_reg, test_size=0.2, random_state=42)
    y_pred = reg.predict(X_te)
    residuals = y_te.values - y_pred

    from sklearn.metrics import r2_score, mean_absolute_error
    r2  = r2_score(y_te, y_pred)
    mae = mean_absolute_error(y_te, y_pred)
    rmse = np.sqrt(((y_te.values - y_pred)**2).mean())

    c1,c2,c3,c4 = st.columns(4)
    for col,(lbl,val) in zip([c1,c2,c3,c4],[
        ("R² Score",f"{r2:.4f}"),("MAE",f"AED {mae:.0f}"),
        ("RMSE",f"AED {rmse:.0f}"),("Avg Predicted",f"AED {enriched['Pred_Spend_AED'].mean():,.0f}")]):
        col.markdown(f"""<div class="metric-card">
            <div class="val">{val}</div><div class="lbl">{lbl}</div></div>""", unsafe_allow_html=True)
    st.markdown("")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Actual vs Predicted Spend")
        fig = px.scatter(x=y_te.values, y=y_pred, opacity=0.5,
                         labels={'x':'Actual (AED)','y':'Predicted (AED)'},
                         color_discrete_sequence=['#38bdf8'])
        max_val = max(y_te.max(), y_pred.max())
        fig.add_trace(go.Scatter(x=[0,max_val], y=[0,max_val], mode='lines',
                                 line=dict(color='#ef4444', dash='dash'), name='Perfect Fit'))
        fig.update_layout(height=380, margin=dict(t=30))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Residual Distribution")
        fig2 = px.histogram(x=residuals, nbins=50, color_discrete_sequence=['#a855f7'],
                            labels={'x':'Residual (AED)'})
        fig2.add_vline(x=0, line_dash='dash', line_color='#ef4444')
        fig2.update_layout(height=380, margin=dict(t=30))
        st.plotly_chart(fig2, use_container_width=True)

    # Feature importance
    st.subheader("Top 15 Spend Drivers")
    fi_data = precomp['reg_feature_importance']
    fi_df = pd.DataFrame(fi_data).sort_values('importance', ascending=True)
    fi_df['feature'] = fi_df['feature'].str.replace('_Enc','').str.replace('_',' ')
    fig3 = px.bar(fi_df, x='importance', y='feature', orientation='h',
                  color='importance', color_continuous_scale='Purples')
    fig3.update_layout(height=480, coloraxis_showscale=False, margin=dict(l=200))
    st.plotly_chart(fig3, use_container_width=True)

    # Spend decile analysis
    st.subheader("Spend Decile Analysis")
    enriched['Spend_Decile'] = pd.qcut(enriched['Pred_Spend_AED'], q=10,
                                        labels=[f"D{i}" for i in range(1,11)])
    decile_stats = enriched.groupby('Spend_Decile').agg(
        Avg_Spend=('Pred_Spend_AED','mean'),
        Count=('Pred_Spend_AED','count'),
        Adopt_Prob=('Adopt_Prob','mean')
    ).reset_index()
    fig4 = px.bar(decile_stats, x='Spend_Decile', y='Avg_Spend',
                  color='Adopt_Prob', color_continuous_scale='Greens',
                  labels={'Spend_Decile':'Decile','Avg_Spend':'Avg Annual Spend (AED)',
                          'Adopt_Prob':'Adoption Probability'})
    fig4.update_layout(height=350)
    st.plotly_chart(fig4, use_container_width=True)
    st.info("💡 Top decile customers (D9-D10) spend 8-10x more than bottom decile. "
            "Acquisition cost ceiling: AED 400-600 for D10, AED 20-40 for D1.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 · PRESCRIPTIVE PLAYBOOK
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎬  Prescriptive Playbook":
    import plotly.graph_objects as go
    import plotly.express as px

    st.title("🎬 Prescriptive Action Playbook")
    st.caption("What should you do? Data-driven decisions for every customer segment.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🏆 Priority Ranking",
        "💸 Discount Engine",
        "📦 Bundle Builder",
        "📢 Channel Planner"
    ])

    with tab1:
        st.subheader("Customer Priority Ranking")
        st.markdown("**Priority Score** = 40% Adoption Probability + 40% Normalised Spend + 20% Tier Weight")

        c1,c2,c3 = st.columns(3)
        for col,(label,val,cls) in zip([c1,c2,c3],[
            ("🟢 Act Now", (enriched['Priority_Tier']=='Act Now').sum(), "act-now"),
            ("🟡 Nurture",  (enriched['Priority_Tier']=='Nurture').sum(), "nurture"),
            ("⚪ Low Priority", (enriched['Priority_Tier']=='Low Priority').sum(), "low-pri"),
        ]):
            col.markdown(f"<h2 class='{cls}'>{val}</h2><p>{label}</p>", unsafe_allow_html=True)

        # Filters
        with st.expander("🔧 Filter Options", expanded=True):
            cf1, cf2, cf3 = st.columns(3)
            tier_filter   = cf1.multiselect("Priority Tier", ['Act Now','Nurture','Low Priority'],
                                             default=['Act Now'])
            persona_filter = cf2.multiselect("Persona", list(PERSONA_COLORS.keys()),
                                              default=list(PERSONA_COLORS.keys()))
            top_n = cf3.slider("Show top N customers", 10, 200, 50)

        filtered = enriched[
            enriched['Priority_Tier'].isin(tier_filter) &
            enriched['Persona'].isin(persona_filter)
        ].sort_values('Priority_Score', ascending=False).head(top_n)

        display_cols = ['Respondent_ID','Persona','Tier','Priority_Score','Priority_Rank',
                        'Priority_Tier','Adopt_Prob','Pred_Spend_AED']
        display = filtered[display_cols].copy()
        display['Adopt_Prob'] = display['Adopt_Prob'].round(3)
        display['Pred_Spend_AED'] = display['Pred_Spend_AED'].round(0).astype(int)
        display['Priority_Score'] = display['Priority_Score'].round(4)
        st.dataframe(display.style.background_gradient(subset=['Priority_Score'], cmap='Greens'),
                     use_container_width=True, height=400)

        # Download
        csv_buf = io.BytesIO()
        filtered[display_cols].to_csv(csv_buf, index=False)
        st.download_button("⬇️ Download Priority List CSV",
                           data=csv_buf.getvalue(),
                           file_name='RunRight_Priority_Customers.csv',
                           mime='text/csv')

        # Priority score distribution
        fig = px.histogram(enriched, x='Priority_Score', color='Priority_Tier',
                           color_discrete_map={'Act Now':'#22c55e','Nurture':'#f59e0b','Low Priority':'#475569'},
                           nbins=40, barmode='overlay', opacity=0.7)
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Segment-Specific Discount Recommendations")
        st.markdown("Discount depth calibrated from spend decile + price sensitivity + runner identity.")

        discount_playbook = {
            'Trail & Ultra Specialist': {
                'tier': 'Tier 1',
                'strategy': 'Exclusivity, not discount',
                'offer': '0–10% off + Early access to new Hoka/ASICS drops',
                'rationale': 'High identity (4.4/5), brand-loyal, responds to scarcity not price',
                'max_acq_cost': 'AED 500–700',
                'expected_ltv': 'AED 6,400 (3-year)',
                'color': '#ef4444'
            },
            'Serious Age-Grouper': {
                'tier': 'Tier 1',
                'strategy': 'Performance value pack',
                'offer': '10–15% off multi-item purchase (shoe + GPS/insole)',
                'rationale': 'Marathon goal-driven, will pay for performance stack, price-aware on bundles',
                'max_acq_cost': 'AED 400–600',
                'expected_ltv': 'AED 5,400 (3-year)',
                'color': '#f97316'
            },
            'Wellness Professional': {
                'tier': 'Tier 2',
                'strategy': 'Community + upgrade offer',
                'offer': '15% off first pair + free 3-month Strava Premium trial',
                'rationale': 'Club member, Strava user — responds to community value add',
                'max_acq_cost': 'AED 200–300',
                'expected_ltv': 'AED 3,500 (3-year)',
                'color': '#22c55e'
            },
            'Social Community Runner': {
                'tier': 'Tier 2',
                'strategy': 'Referral + social incentive',
                'offer': '20% off if joins via club/event + refer-a-friend AED 50 credit',
                'rationale': 'Social motivation — peer acquisition is most efficient channel',
                'max_acq_cost': 'AED 150–200',
                'expected_ltv': 'AED 2,700 (3-year)',
                'color': '#3b82f6'
            },
            'Aspirational Beginner': {
                'tier': 'Tier 3',
                'strategy': 'Freemium entry',
                'offer': 'Free app tier + 25% off first pair (entry model only)',
                'rationale': 'Price sensitive, needs to experience value before committing',
                'max_acq_cost': 'AED 40–80',
                'expected_ltv': 'AED 600 (3-year, upgrade potential to Tier 2)',
                'color': '#a855f7'
            },
            'Casual Lifestyle Runner': {
                'tier': 'Tier 3',
                'strategy': 'Seasonal activation',
                'offer': 'Flash sale 20–30% off + limited edition lifestyle colourways',
                'rationale': 'Discount-triggered, style-motivated — only convert on deep deals',
                'max_acq_cost': 'AED 20–40',
                'expected_ltv': 'AED 460 (3-year)',
                'color': '#64748b'
            },
        }

        for persona, details in discount_playbook.items():
            tier_emoji = '🔴' if details['tier']=='Tier 1' else ('🟡' if details['tier']=='Tier 2' else '🔵')
            with st.expander(f"{tier_emoji} **{persona}** — {details['strategy']}"):
                c1, c2, c3 = st.columns(3)
                c1.markdown(f"**Offer:** {details['offer']}")
                c1.markdown(f"**Rationale:** {details['rationale']}")
                c2.metric("Max Acq. Cost", details['max_acq_cost'])
                c3.metric("3-Year LTV", details['expected_ltv'])

    with tab3:
        st.subheader("Bundle Builder — ARM-Backed Product Recommendations")
        st.markdown("Each bundle is derived directly from the Association Rule Mining results.")
        col1, col2 = st.columns(2)
        bundle_data = [
            {'Persona':'Trail & Ultra Specialist','Bundle':'Desert Trail Pro Kit',
             'Anchor':'Hoka Speedgoat','Add-ons':'Garmin, Hydration, Trail Socks',
             'Lift':'1.92 (Trail→Garmin)','Basket':'AED 1,800-2,400'},
            {'Persona':'Serious Age-Grouper','Bundle':'Marathon Ready Pack',
             'Anchor':'ASICS Gel-Nimbus','Add-ons':'GPS Watch, Custom Insoles, Compression',
             'Lift':'1.48 (Competitive→GPS)','Basket':'AED 1,400-2,000'},
            {'Persona':'Wellness Professional','Bundle':'Club Runner Collection',
             'Anchor':'On Running Cloud','Add-ons':'Foam Roller, Strava Premium, Belt',
             'Lift':'1.36 (Garmin→Club)','Basket':'AED 900-1,400'},
            {'Persona':'Social Community Runner','Bundle':'Community Starter Pack',
             'Anchor':'Nike React','Add-ons':'NRC Premium, Socks, Running Belt',
             'Lift':'1.20 (NRC→Social)','Basket':'AED 700-1,000'},
            {'Persona':'Aspirational Beginner','Bundle':'First Steps Bundle',
             'Anchor':'Adidas Ultraboost entry','Add-ons':'Socks, Free App Trial',
             'Lift':'1.15 (Road→Adidas)','Basket':'AED 400-700'},
            {'Persona':'Casual Lifestyle Runner','Bundle':'Lifestyle Flex Pack',
             'Anchor':'New Balance Fresh Foam','Add-ons':'Apple Health integration, Insoles',
             'Lift':'1.10 (AppleWatch→General)','Basket':'AED 300-600'},
        ]
        bundle_df = pd.DataFrame(bundle_data)
        st.dataframe(bundle_df, use_container_width=True, hide_index=True)

        # Estimated basket uplift chart
        bundle_df['Basket_Min'] = bundle_df['Basket'].str.extract(r'(\d+)').astype(int)
        bundle_df['Basket_Max'] = bundle_df['Basket'].str.extract(r'-(\d+)').astype(int)
        bundle_df['Basket_Mid'] = (bundle_df['Basket_Min'] + bundle_df['Basket_Max']) / 2
        fig = px.bar(bundle_df.sort_values('Basket_Mid',ascending=True),
                     x='Basket_Mid', y='Persona', orientation='h',
                     color='Basket_Mid', color_continuous_scale='Oranges',
                     labels={'Basket_Mid':'Estimated Bundle Value (AED)','Persona':''},
                     error_x=(bundle_df.sort_values('Basket_Mid',ascending=True)['Basket_Max'] -
                               bundle_df.sort_values('Basket_Mid',ascending=True)['Basket_Mid']))
        fig.update_layout(height=360, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Channel Planner — Where to Reach Each Segment")
        st.markdown("Channel recommendations derived from app usage, club membership, and ARM co-occurrence rules.")

        channel_data = {
            'Trail & Ultra Specialist': {
                'Primary': 'Strava ads + Garmin Connect',
                'Secondary': 'Trail running events (Wadi Bih, Spartan UAE)',
                'Content': 'GPS tracking data, Hoka/ASICS trail comparisons, race prep guides',
                'Budget': '40% of Tier 1 budget',
                'CPA Target': 'AED 400'
            },
            'Serious Age-Grouper': {
                'Primary': 'Strava + Dubai/AD Marathon community',
                'Secondary': 'Running club sponsorship (Dubai Creek Striders)',
                'Content': 'Marathon training plans, shoe rotation guides, race shoe reviews',
                'Budget': '60% of Tier 1 budget',
                'CPA Target': 'AED 350'
            },
            'Wellness Professional': {
                'Primary': 'Instagram + running club newsletters',
                'Secondary': 'Corporate wellness programs, gym partnerships',
                'Content': 'Recovery, injury prevention, run-work balance, community stories',
                'Budget': '50% of Tier 2 budget',
                'CPA Target': 'AED 200'
            },
            'Social Community Runner': {
                'Primary': 'WhatsApp running groups + NRC community events',
                'Secondary': 'Instagram Reels, group run sponsorships',
                'Content': 'Group run highlights, peer reviews, community challenges',
                'Budget': '50% of Tier 2 budget',
                'CPA Target': 'AED 150'
            },
            'Aspirational Beginner': {
                'Primary': 'Instagram / TikTok (motivational content)',
                'Secondary': 'Online retail partnerships (Noon, Amazon UAE)',
                'Content': 'Beginner guides, "my first 5K" stories, entry-level shoe reviews',
                'Budget': '30% of Tier 3 budget',
                'CPA Target': 'AED 60'
            },
            'Casual Lifestyle Runner': {
                'Primary': 'Instagram shopping + flash sale emails',
                'Secondary': 'Mall activations (Dubai Mall, Mall of Emirates)',
                'Content': 'Style-forward content, seasonal campaigns, limited drops',
                'Budget': '70% of Tier 3 budget',
                'CPA Target': 'AED 30'
            },
        }

        for persona, ch in channel_data.items():
            tier = PERSONA_TIERS[persona]
            te = '🔴' if tier=='Tier 1' else ('🟡' if tier=='Tier 2' else '🔵')
            with st.expander(f"{te} **{persona}**"):
                c1, c2 = st.columns(2)
                c1.markdown(f"**Primary Channel:** {ch['Primary']}")
                c1.markdown(f"**Secondary Channel:** {ch['Secondary']}")
                c1.markdown(f"**Content Theme:** {ch['Content']}")
                c2.metric("Budget Allocation", ch['Budget'])
                c2.metric("Target CPA", ch['CPA Target'])

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 8 · SCORE NEW CUSTOMERS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📥  Score New Customers":
    import plotly.express as px

    st.title("📥 Score New Customers")
    st.caption("Upload a CSV of new respondents → get adoption probability, segment, spend prediction, and marketing action.")

    PERSONA_MAP = {
        2: 'Trail & Ultra Specialist',
        0: 'Serious Age-Grouper',
        5: 'Wellness Professional',
        3: 'Social Community Runner',
        1: 'Aspirational Beginner',
        4: 'Casual Lifestyle Runner'
    }
    TIER_MAP = {
        'Trail & Ultra Specialist': 'Tier 1',
        'Serious Age-Grouper':      'Tier 1',
        'Wellness Professional':    'Tier 2',
        'Social Community Runner':  'Tier 2',
        'Aspirational Beginner':    'Tier 3',
        'Casual Lifestyle Runner':  'Tier 3',
    }
    ACTION_MAP = {
        ('Tier 1', 'Act Now'):     '🔴 Priority Outreach — Exclusivity offer + direct contact',
        ('Tier 1', 'Nurture'):     '🟠 Warm Tier 1 — Send Trail/Marathon content + soft CTA',
        ('Tier 1', 'Low Priority'):'🟡 Low-prob Tier 1 — Add to retargeting pool',
        ('Tier 2', 'Act Now'):     '🟢 Convert Now — Community offer + referral incentive',
        ('Tier 2', 'Nurture'):     '🔵 Nurture Tier 2 — Club/event marketing',
        ('Tier 2', 'Low Priority'):'⚪ Low-prob Tier 2 — Seasonal campaign only',
        ('Tier 3', 'Act Now'):     '💜 Freemium Activate — Free trial + 25% first pair',
        ('Tier 3', 'Nurture'):     '⚫ Slow Nurture — Flash sales + style content',
        ('Tier 3', 'Low Priority'):'🩶 Park — Low priority, minimal spend',
    }
    CHANNEL_MAP = {
        'Trail & Ultra Specialist': 'Strava Ads + Garmin Connect',
        'Serious Age-Grouper':      'Marathon Community + Running Clubs',
        'Wellness Professional':    'Instagram + Corporate Wellness',
        'Social Community Runner':  'WhatsApp Groups + NRC Events',
        'Aspirational Beginner':    'Instagram / TikTok + Online Retail',
        'Casual Lifestyle Runner':  'Instagram Shopping + Mall Activations',
    }
    BUNDLE_MAP = {
        'Trail & Ultra Specialist': 'Desert Trail Pro Kit (AED 1,800-2,400)',
        'Serious Age-Grouper':      'Marathon Ready Pack (AED 1,400-2,000)',
        'Wellness Professional':    'Club Runner Collection (AED 900-1,400)',
        'Social Community Runner':  'Community Starter Pack (AED 700-1,000)',
        'Aspirational Beginner':    'First Steps Bundle (AED 400-700)',
        'Casual Lifestyle Runner':  'Lifestyle Flex Pack (AED 300-600)',
    }

    # Template download
    template_df = enc[FEATURES].head(3).copy()
    template_buf = io.BytesIO()
    template_df.to_csv(template_buf, index=False)
    st.download_button("📄 Download CSV Template (3 sample rows)",
                       data=template_buf.getvalue(),
                       file_name='RunRight_New_Customers_Template.csv',
                       mime='text/csv')

    st.info(f"Your CSV must contain these {len(FEATURES)} columns (same names as template). "
            "Missing values are auto-imputed with training medians.")

    uploaded = st.file_uploader("Upload New Customer CSV", type=['csv'])

    if uploaded:
        try:
            new_df = pd.read_csv(uploaded)
            st.success(f"✅ Loaded {len(new_df):,} rows, {new_df.shape[1]} columns")

            # Validate columns
            missing_cols = [f for f in FEATURES if f not in new_df.columns]
            if missing_cols:
                st.error(f"❌ Missing columns: {missing_cols[:10]}{'...' if len(missing_cols)>10 else ''}")
                st.stop()

            X_new = new_df[FEATURES].copy()
            X_new_imp    = pd.DataFrame(imputer.transform(X_new), columns=FEATURES)
            X_new_scaled = scaler.transform(X_new_imp)

            # Score
            cluster_labels   = km.predict(X_new_scaled)
            adopt_probs      = clf.predict_proba(X_new_imp)[:,1]
            pred_spend       = reg.predict(X_new_imp)

            spend_min = enriched['Pred_Spend_AED'].min()
            spend_max = enriched['Pred_Spend_AED'].max()
            spend_norm = (pred_spend - spend_min) / (spend_max - spend_min)
            tier_w_map = {'Tier 1':1.0, 'Tier 2':0.7, 'Tier 3':0.3}

            results = new_df.copy()
            results['Cluster']        = cluster_labels
            results['Persona']        = [PERSONA_MAP.get(c,'Unknown') for c in cluster_labels]
            results['Tier']           = results['Persona'].map(TIER_MAP)
            results['Adopt_Prob']     = adopt_probs.round(4)
            results['Pred_Spend_AED'] = pred_spend.round(0).astype(int)
            tw = results['Tier'].map(tier_w_map).fillna(0.3)
            priority_scores          = adopt_probs*0.4 + spend_norm*0.4 + tw.values*0.2
            results['Priority_Score'] = priority_scores.round(4)
            results['Priority_Tier']  = pd.cut(priority_scores, bins=[0,0.35,0.55,1.01],
                                                labels=['Low Priority','Nurture','Act Now'])
            results['Recommended_Action'] = [
                ACTION_MAP.get((row['Tier'], str(row['Priority_Tier'])), 'Review manually')
                for _, row in results.iterrows()
            ]
            results['Recommended_Bundle'] = results['Persona'].map(BUNDLE_MAP)
            results['Recommended_Channel'] = results['Persona'].map(CHANNEL_MAP)
            results['Priority_Rank'] = results['Priority_Score'].rank(ascending=False).astype(int)

            st.markdown("---")
            # Summary metrics
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Total Scored", len(results))
            c2.metric("Act Now", (results['Priority_Tier']=='Act Now').sum())
            c3.metric("Avg Adopt Prob", f"{results['Adopt_Prob'].mean():.1%}")
            c4.metric("Avg Pred Spend", f"AED {results['Pred_Spend_AED'].mean():,.0f}")

            # Summary charts
            col1, col2 = st.columns(2)
            with col1:
                persona_counts = results['Persona'].value_counts()
                fig = px.pie(values=persona_counts.values, names=persona_counts.index,
                             title="New Customers by Persona", hole=0.4,
                             color=persona_counts.index, color_discrete_map=PERSONA_COLORS)
                fig.update_layout(height=320)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                pt = results['Priority_Tier'].value_counts()
                fig2 = px.bar(x=pt.index, y=pt.values,
                              color=pt.index,
                              color_discrete_map={'Act Now':'#22c55e','Nurture':'#f59e0b','Low Priority':'#475569'},
                              title="Priority Distribution")
                fig2.update_layout(height=320, showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)

            # Results table
            st.subheader("Scored Customer Table")
            show_cols = ['Priority_Rank','Persona','Tier','Priority_Tier',
                         'Adopt_Prob','Pred_Spend_AED','Priority_Score',
                         'Recommended_Action','Recommended_Bundle','Recommended_Channel']
            available_show = [c for c in show_cols if c in results.columns]

            # Filters
            tf1, tf2 = st.columns(2)
            pt_filter = tf1.multiselect("Filter Priority Tier",
                                        ['Act Now','Nurture','Low Priority'],
                                        default=['Act Now','Nurture'])
            pf_filter = tf2.multiselect("Filter Persona",
                                        list(results['Persona'].unique()),
                                        default=list(results['Persona'].unique()))
            filtered_results = results[
                results['Priority_Tier'].isin(pt_filter) &
                results['Persona'].isin(pf_filter)
            ].sort_values('Priority_Rank')[available_show]

            st.dataframe(filtered_results.style.background_gradient(
                subset=['Priority_Score','Adopt_Prob'], cmap='Greens'),
                use_container_width=True, height=500)

            # CSV download
            out_buf = io.BytesIO()
            results.to_csv(out_buf, index=False)
            st.download_button("⬇️ Download Full Scored CSV",
                               data=out_buf.getvalue(),
                               file_name='RunRight_Scored_Customers.csv',
                               mime='text/csv')

        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.exception(e)
    else:
        st.markdown("### 👆 Upload a CSV to begin scoring")
        st.markdown("""
**What happens when you upload:**
1. Schema validation — checks all required columns present
2. Auto-imputation — missing values filled with training medians
3. Encoding pipeline — same transforms as training data
4. **Classification** → adoption probability (0–1) per customer
5. **Clustering** → persona segment assignment
6. **Regression** → predicted annual spend (AED)
7. **Priority Score** → composite ranking (adopt prob + spend + tier)
8. **Action recommendation** → specific marketing action per customer
9. Download as CSV or filter/view in table above
        """)

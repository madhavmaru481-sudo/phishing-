import re, joblib, streamlit as st

vec = joblib.load("models/vec.joblib")
model = joblib.load("models/model.joblib")
names = vec.get_feature_names_out()

def clean(t):
    t = str(t).lower()
    t = re.sub(r"(http\S*|www\.\S+)", " urltoken ", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

RED_FLAGS = {
    "Urgency language": r"\b(urgent|immediately|within 24 hours|act now|verify now|final notice|expires?)\b",
    "Account threat": r"\b(suspended|locked|deactivated|unauthorized|compromised|blocked)\b",
    "Credential request": r"\b(password|login|verify your|confirm your|otp|pin|ssn|cvv)\b",
    "Money / prize lure": r"\b(winner|prize|lottery|refund|claim|gift card|reward)\b",
    "Shortened URL": r"(bit\.ly|tinyurl|goo\.gl|t\.co|cutt\.ly)",
    "IP-based URL": r"https?://\d{1,3}(\.\d{1,3}){3}",
}

def red_flags(text):
    found = [n for n, pat in RED_FLAGS.items() if re.search(pat, text, re.I)]
    if len(re.findall(r"https?://", text)) >= 3:
        found.append("Many links")
    return found

SAMPLES = {
    "-- choose sample --": "",
    "Phishing: bank": "Dear customer, your account has been suspended. Verify your password immediately at http://192.168.4.12/secure-login or it will be permanently blocked within 24 hours.",
    "Phishing: prize": "Congratulations! You are the winner of a $1000 gift card. Claim your reward now: http://bit.ly/claim-now",
    "Legit: office": "Hi team, the project meeting is moved to 3 PM tomorrow. Please review the attached agenda before joining. Thanks, Rahul",
}

st.title("🎣 Phishing / Social Engineering Detector")
st.caption("Hybrid: NLP (TF-IDF) + Logistic Regression + rule-based red flags")

choice = st.selectbox("Sample message", list(SAMPLES.keys()))
text = st.text_area("Paste email / SMS", value=SAMPLES[choice], height=200)

if st.button("Analyze") and text.strip():
    x = vec.transform([clean(text)])
    p = model.predict_proba(x)[0][1]          # model score
    flags = red_flags(text)
    rule = min(len(flags) / 4, 1.0)           # rule score
    final = 0.7 * p + 0.3 * rule              # combined risk score

    if final < 0.30:
        st.success(f"✅ SAFE  (risk score: {final:.0%})")
    elif final < 0.60:
        st.warning(f"⚠️ SUSPICIOUS  (risk score: {final:.0%})")
    else:
        st.error(f"🚨 PHISHING  (risk score: {final:.0%})")
    st.progress(float(final))

    m1, m2 = st.columns(2)
    m1.metric("Model score (ML)", f"{p:.0%}")
    m2.metric("Red-flag score (rules)", f"{rule:.0%}")
    st.caption("Risk score = 70% model score + 30% red-flag score")

    contrib = x.multiply(model.coef_[0]).tocoo()
    pairs = sorted(zip(contrib.col, contrib.data), key=lambda t: -t[1])
    top = [(names[i], round(float(v), 3)) for i, v in pairs[:8] if v > 0]

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Words pushing towards phishing")
        if top:
            for w, v in top:
                st.write(f"• **{w}**  (+{v})")
        else:
            st.write("No strong phishing words found.")
    with c2:
        st.subheader("Red flags detected")
        if flags:
            for f in flags:
                st.write(f"⚠️ {f}")
        else:
            st.write("No rule-based red flags.")
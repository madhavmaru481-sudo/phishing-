import re, joblib, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def clean(t):
    t = str(t).lower()
    t = re.sub(r"(http\S*|www\.\S+)", " urltoken ", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

# ---------- Dataset A: emails ----------
a = pd.read_csv("data/Phishing_Email.csv")
a = a.dropna(subset=["Email Text", "Email Type"])
a = a[a["Email Text"].str.strip().str.lower() != "empty"]
a = a.drop_duplicates(subset="Email Text")
A = pd.DataFrame({
    "text": a["Email Text"],
    "y": (a["Email Type"] == "Phishing Email").astype(int),
    "src": "email",
})

# ---------- Dataset B: SMS ----------
b = pd.read_csv("data/spam.csv", encoding="latin-1")[["v1", "v2"]]
b.columns = ["label", "text"]
b = b.dropna().drop_duplicates(subset="text")
B = pd.DataFrame({
    "text": b["text"],
    "y": (b["label"] == "spam").astype(int),
    "src": "sms",
})

A["clean"] = A["text"].apply(clean)
B["clean"] = B["text"].apply(clean)
print("Emails:", len(A), " SMS:", len(B))

def new_vec():
    return TfidfVectorizer(ngram_range=(1, 2), max_features=50000,
                           sublinear_tf=True, stop_words="english")

def new_model():
    return LogisticRegression(max_iter=1000, class_weight="balanced")

# ---------- Test 1: cross-dataset (honest test) ----------
def cross_test(train_df, test_df, title):
    v = new_vec()
    m = new_model().fit(v.fit_transform(train_df["clean"]), train_df["y"])
    pred = m.predict(v.transform(test_df["clean"]))
    print("\n=====", title, "=====")
    print(classification_report(test_df["y"], pred,
          target_names=["Safe", "Phishing/Spam"], zero_division=0))
    print(confusion_matrix(test_df["y"], pred))

cross_test(A, B, "Train on EMAIL, test on SMS (never seen)")
cross_test(B, A, "Train on SMS, test on EMAIL (never seen)")

# ---------- Final model: merged data ----------
df = pd.concat([A, B], ignore_index=True)
strat = df["src"] + df["y"].astype(str)
tr, te = train_test_split(df, test_size=0.2, stratify=strat, random_state=42)

vec = new_vec()
Xtr = vec.fit_transform(tr["clean"])
model = new_model().fit(Xtr, tr["y"])

print("\n===== FINAL MODEL (merged data) =====")
pred = model.predict(vec.transform(te["clean"]))
print(classification_report(te["y"], pred, target_names=["Safe", "Phishing/Spam"]))
print(confusion_matrix(te["y"], pred))

for s in ["email", "sms"]:
    part = te[te["src"] == s]
    p = model.predict(vec.transform(part["clean"]))
    print(f"\n--- Only {s.upper()} test rows ---")
    print(classification_report(part["y"], p, target_names=["Safe", "Phishing/Spam"]))

names = vec.get_feature_names_out()
top = model.coef_[0].argsort()[-20:][::-1]
print("Top phishing words:", [names[i] for i in top])

joblib.dump(vec, "models/vec.joblib")
joblib.dump(model, "models/model.joblib")
print("Saved model files.")
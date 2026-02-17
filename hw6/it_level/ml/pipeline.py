from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

from it_level.settings import Settings


def make_sklearn_pipeline(s: Settings) -> Pipeline:
    num_cols = ["salary_rub", "age", "gender", "exp_months"]
    cat_cols = ["city"]

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            (
                "text_tfidf",
                TfidfVectorizer(
                    max_features=s.max_tfidf_features,
                    min_df=2,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    norm="l2",
                    lowercase=True,
                ),
                "text",
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    clf = LinearSVC(
        class_weight="balanced",
        C=1.0,
    )

    return Pipeline([("pre", pre), ("clf", clf)])

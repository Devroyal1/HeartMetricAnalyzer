# type: ignore
import warnings
import io
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# classification packages from scikit-learn ~ sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# for required data and content for the models
#from content_switcher import content_switcher
content_switcher = {
    "Naive Bayes": "Naive Bayes is a classification technique based on Bayes' Theorem with an assumption of independence among predictors. A Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature.",
    "K-Nearest Neighbors": "K-Nearest Neighbors is a simple, easy-to-implement supervised machine learning algorithm that can be used to solve both classification and regression problems.",
    "Support Vector Machines": "Support Vector Machines is a supervised machine learning algorithm that can be used for both classification or regression challenges. However, it is mostly used in classification problems.",
    "Multilayer Perceptron": "Multilayer Perceptron is a class of feedforward artificial neural network. An MLP consists of at least three layers of nodes: an input layer, a hidden layer, and an output layer.",
    "Gradient Boosting": "Gradient Boosting is a machine learning technique for regression and classification problems, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees.",
    "Random Forest": "Random Forest is an ensemble learning method for classification, regression, and other tasks that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes of the individual trees.",
    "Decision Tree": "Decision Tree is a flowchart-like tree structure where an internal node represents a feature(or attribute), the branch represents a decision rule, and each leaf node represents the outcome.",
    "Logistic Regression": "Logistic Regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable."
}
#from footer import footer

warnings.filterwarnings('ignore')

try:
    def display_results(model_name, y_prediction):
        st.subheader(model_name)
        st.write(content_switcher[model_name])

        cm = confusion_matrix(Y_test, y_prediction)
        accuracy = round(accuracy_score(Y_test, y_prediction) * 100, 2)
        precision = round(precision_score(Y_test, y_prediction) * 100, 2)
        recall = round(recall_score(Y_test, y_prediction) * 100, 2)
        f1 = round(f1_score(Y_test, y_prediction, average='weighted') * 100, 2)

        st.html("<b><i>Confusion Matrix: </i></b>")
        st.write(
            pd.DataFrame({
                "0": cm[:, 0],
                "1": cm[:, 1]
            })
        )

        # st.metric() API
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{accuracy}%")
        col2.metric("Precision", f"{precision}%")
        col3.metric("Recall", f"{recall}%")
        col4.metric("F1-Score", f"{f1}%")
        st.divider()
        return accuracy, precision, recall, f1

    title_of_the_project = "Data Science Approaches for Heart Disease Prediction: A Comparative Study of Classification Algorithms"

    st.title(title_of_the_project.title())
    st.sidebar.title(title_of_the_project.title())
    st.sidebar.subheader("Settings")

    # upload the dataset
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        dataset = pd.read_csv(uploaded_file)
    else:
        st.error("Please upload a CSV file.")
        #footer()
        st.stop()

    # display basic information about the dataset
    col1, col2, col3 = st.columns([.5, 7.5, .5], vertical_alignment="center")
    with col2:
        st.subheader("Dataset Information")

        st.write("Dataset head:")
        st.write(dataset.head())
        st.divider()

        st.write("Dataset description:")
        st.write(dataset.describe())
        st.divider()

        st.write("The complete data provided by the dataset:")
        st.write(dataset)
        st.divider()

        # Display dataset info
        buffer = io.StringIO()
        dataset.info(buf=buffer)
        s = buffer.getvalue()

        st.write("Dataset attributes with it's data type and memory usages:")
        st.text(s)
        st.divider()

        # Target distribution
        st.subheader("Target Distribution")
        st.html('''
            <details>
                <summary>About</summary>
                The target variable 'cardio' is the presence or absence of heart disease. The value 1 represents the presence of heart disease and the value 0 represents the absence of heart disease.
            </details>
        ''')
        fig, ax = plt.subplots()
        sns.countplot(x=dataset["cardio"], data=dataset, ax=ax)
        st.pyplot(fig)

    st.divider()

    # Split dataset
    predictors = dataset.drop("cardio", axis=1)
    target = dataset["cardio"]
    X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.20, random_state=0)

    # Model Building and Evaluation 
    st.write("## Model Building and Evaluation")

    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, Y_train)
    Y_prediction_nb = nb.predict(X_test)
    accuracy_nb, precision_nb, recall_nb, f1_nb = display_results("Naive Bayes", Y_prediction_nb)

    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(X_train, Y_train)
    Y_prediction_knn = knn.predict(X_test)
    accuracy_knn, precision_knn, recall_knn, f1_knn = display_results("K-Nearest Neighbors", Y_prediction_knn)

    # Support Vector Machines - Linear
    svm_model = svm.LinearSVC()
    svm_model.fit(X_train, Y_train)
    Y_prediction_svm = svm_model.predict(X_test)
    accuracy_svm, precision_svm, recall_svm, f1_svm = display_results("Support Vector Machines", Y_prediction_svm)

    # Neural Networks - MLP
    mlp = MLPClassifier(hidden_layer_sizes=(90,), max_iter=1000)
    mlp.fit(X_train, Y_train)
    Y_pred_mlp = mlp.predict(X_test)
    accuracy_mlp, precision_mlp, recall_mlp, f1_mlp = display_results("Multilayer Perceptron", Y_pred_mlp)

    # Gradient Boosting - the best for now
    gb = GradientBoostingClassifier()
    gb.fit(X_train, Y_train)
    Y_pred_gb = gb.predict(X_test)
    accuracy_gb, precision_gb, recall_gb, f1_gb = display_results("Gradient Boosting", Y_pred_gb)

    # Random Forest - n_estimators = 600 and random_state = 42 - default_since
    rf = RandomForestClassifier(n_estimators=600, random_state=42)
    rf.fit(X_train, Y_train)
    Y_pred_rf = rf.predict(X_test)
    accuracy_rf, precision_rf, recall_rf, f1_rf = display_results("Random Forest", Y_pred_rf)

    # Decision Tree
    dt = DecisionTreeClassifier()
    dt.fit(X_train, Y_train)
    Y_pred_dt = dt.predict(X_test)
    accuracy_dt, precision_dt, recall_dt, f1_dt = display_results("Decision Tree", Y_pred_dt)

    # Logistic Regression
    # lr = LogisticRegression()
    # lr.fit(X_train, Y_train)
    # Y_pred_lr = lr.predict(X_test)
    # accuracy_lr, precision_lr, recall_lr, f1_lr = display_results("Logistic Regression", Y_pred_lr)

    # Bar plots for categorical features
    st.subheader("Categorical Features vs cardio")
    st.html('''
        <details>
            <summary>About</summary>
            Below are the bar plots showing the relationship between categorical features and the target variable 'cardio'. These plots help in understanding how different categorical features are distributed with respect to the presence or absence of heart disease.
        </details>
    ''')
    categorical_features = ["cholesterol", "gluc", "smoke", "alco", "active"]
    for feature in categorical_features:
        fig, ax = plt.subplots()
        sns.barplot(x=feature, y=dataset["cardio"], data=dataset, ax=ax)
        st.pyplot(fig)

    st.write('''
        ## Model Comparison
        Below are the comparison of all the models based on the metrics such as Accuracy, Precision, Recall and F1-Score.
    ''')


    algorithms = ["Naive Bayes", "K-Nearest Neighbors", "Support Vector Machines", "Multilayer Perceptron", "Gradient Boosting", "Random Forest", "Decision Tree"]

    accuracy_scores = [accuracy_nb, accuracy_knn, accuracy_svm, accuracy_mlp, accuracy_gb, accuracy_rf, accuracy_dt]

    precision_scores = [precision_nb, precision_knn, precision_svm, precision_mlp, precision_gb, precision_rf, precision_dt]

    recall_scores = [recall_nb, recall_knn, recall_svm, recall_mlp, recall_gb, recall_rf, recall_dt]

    f1_scores = [f1_nb, f1_knn, f1_svm, f1_mlp, f1_gb, f1_rf, f1_dt]

    # Plot Accuracy
    st.subheader("Accuracy Score Comparison")
    accuracy_df = pd.DataFrame({
        "Algorithms": algorithms,
        "Accuracy Score": accuracy_scores
    })
    st.bar_chart(accuracy_df.set_index("Algorithms"))
    st.divider()

    # Plot Precision
    st.subheader("Precision Score Comparison")
    precision_df = pd.DataFrame({
        "Algorithms": algorithms,
        "Precision Score": precision_scores
    })
    st.bar_chart(precision_df.set_index("Algorithms"))
    st.divider()

    # Plot Recall
    st.subheader("Recall Score Comparison")
    recall_df = pd.DataFrame({
        "Algorithms": algorithms,
        "Recall Score": recall_scores
    })
    st.bar_chart(recall_df.set_index("Algorithms"))
    st.divider()

    # Plot F1-Score
    st.subheader("F1-Score Comparison")
    f1_df = pd.DataFrame({
        "Algorithms": algorithms,
        "F1-Score": f1_scores
    })
    st.bar_chart(f1_df.set_index("Algorithms"))
    st.divider()

    # Display DataFrame with results
    data = {
        "Algorithm": algorithms,
        "Accuracy": accuracy_scores,
        "Precision": precision_scores,
        "Recall": recall_scores,
        "F1": f1_scores
    }

    df = pd.DataFrame(data)
    df.set_index("Algorithm", inplace=True)
    st.subheader("Comparison of all Performance Metrics")
    st.write(df)
    st.divider()

    st.subheader("Comparison of all Performance Metrics in a single chart")
    st.bar_chart(df, stack=False)

    # Classification Report for Gradient Boosting Model    
    # st.subheader("Gradient Boosting Model")
    report = classification_report(Y_test, Y_pred_gb)
    print(report)

    #footer()
except Exception as e:
    st.error(f"An error occurred: {e}")
    #footer()
    st.stop()
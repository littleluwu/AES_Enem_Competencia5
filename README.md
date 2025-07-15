# 📝 **Automatic Essay Scoring on ENEM: Competencia 5**

This project focuses on evaluating essays following the grading process of ENEM (Exame Nacional do Ensino Médio - National High School Exam), with a specific emphasis on Competence 5. The project was elaborated and developed as part of an assignment for the class `1001305 - PROCESSAMENTO DE LINGUAGEM NATURAL`, lectured by the professor **Tiago A. Almeida** on the **Federal University of São Carlos (UFSCar)**.

## 📋 **About the data**

The dataset consists of essays extracted from the ENEM, along with their respective motivational texts and debate themes. Each essay is structured into paragraphs and stored as a list, however, there is some noise within the essays and how they were broken down into paragraphs. The data also includes essay titles, a score for each competence analyzed in the grading process, and the total scores combining the competencies' scores, respectively.

On the competence scores, each competence can be graded up to 200 points. A key feature of the grading process is that the scores are not continuous; instead, the essays are categorized into specific groups with particular restrictions and requirements. Each group translates to a numerical score to compose the final score when combining the competencies. This characteristic of the data allows the problem to be faced not only as a regression task but also as a classification task.

## 📍 **Proposed Approaches**

In the grading process, competence 5 expects the student to propose a course of action as an intervention method to tackle the problems described throughout the text. To create a clear course of action for these interventions, some key aspects must be addressed. These include identifying the actor responsible for the actions, outlining the actions themselves, explaining how these actions will be carried out, and specifying the desired outcomes.

A proposed approach for developing a solution to this problem is to follow the human grading process, analyzing this information and aspects from the essay in order to evaluate the intervention method the student elaborated. With common patterns observed in the structure of the language and also found in guides and essay models aimed at the ENEM, we can extract this information using rules to filter out the relevant sentences and paragraphs from the text. To understand and locate the actor responsible for the course of action, a pre-processing of Named Entity Recognition (NER) is used to determine where the student refers to an external entity or a specific actor in the essay.

## 🧪 **Obtained Results**

## 📑 **References**

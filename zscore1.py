import pandas as pd
Univ1=pd.read_excel(r"C:\DS\University_Clustering.xlsx")
Univ1
import scipy.stats as stats
un= Univ1['SFRatio']
stats.zscore(Univ1['SFRatio'])
import numpy as np
threshold = 3.0
z_scores = (Univ1['SFRatio']-np.mean(Univ1['SFRatio'])) / np.std(Univ1['SFRatio'])
outliers = np.where(np.abs(z_scores) > threshold)
outliers1= un.iloc[outliers]
print("Outliers:",outliers1)


#A medical trial  is conducted to test whether or not a new medicine reduces cholesterol by 25 percent. State the null and alternative hypotheses.
#We want to test whether the mean GPA of students in American colleges is different from 2.0 (out of 4.0). The null and alternative hypotheses are the following:
#We want to test whether the mean height of eighth graders is 66 inches. State the null and alternative hypotheses. Fill in the correct symbol (=, ≠, ≥, <, ≤, >) for the null and alternative hypotheses.
#We want to test if college students take fewer than five years to graduate from college, on the average. The null and alternative hypotheses are the following:
#We want to test if it takes fewer than 45 minutes to teach a lesson plan. State the null and alternative hypotheses. Fill in the correct symbol ( =, ≠, ≥, <, ≤, >) for the null and alternative hypotheses.
#An article on school standards stated that about half of all students in France, Germany, and Israel take advanced placement exams and a third of the students pass. The same article stated that 6.6 percent of U.S. students take advanced placement exams and 4.4 percent pass. Test if the percentage of U.S. students who take advanced placement exams is more than 6.6 percent. State the null and alternative hypotheses.
#On a state driver’s test, about 40 percent pass the test on the first try. We want to test if more than 40 percent pass on the first try. Fill in the correct symbol (=, ≠, ≥, <, ≤, >) for the null and alternative hypotheses.


#In your clinical study, you compare the symptoms of patients who received the new drug intervention or a control treatment. Using a t test, you obtain a p value of .035. This p value is lower than your alpha of .05, so you consider your results statistically significant and reject the null hypothesis.
#However, the p value means that there is a 3.5% chance of your results occurring if the null hypothesis is true. Therefore, there is still a risk of making a Type I error

#Suppose that building a cafeteria entails profits if more that 40 percent of the students make a purchase (interested = would purchase) a meal plan. Which is more serious
#1) Type 1 error: lose the opportunity to make profits?
#2) Type 2 error: bear the cost and the loss if a cafetria is built?

#Example 1: Medical Testing
#Suppose a new medical test is designed to detect a particular disease. The null hypothesis (H0) is that a person is healthy, and the alternative hypothesis (H1) is that a person has the disease. A Type 1 error would occur if the test incorrectly indicates that a healthy person has the disease. This might lead to unnecessary stress, further testing, or treatment for the person who is actually healthy.
#Example 2: Legal System
#In a criminal trial, the null hypothesis (H0) is that the defendant is innocent, and the alternative hypothesis (H1) is that the defendant is guilty. A Type 1 error would occur if the jury wrongly convicts an innocent person, leading to a miscarriage of justice

#Example 1: Medical Testing
#Continuing with the medical test example, a Type 2 error would occur if the test incorrectly indicates that a person is healthy when they actually have the disease. This might delay necessary treatment and allow the disease to progress.
#Example 2: Quality Control
#In manufacturing, quality control tests are performed to determine whether a product is defective (null hypothesis, H0) or not defective (alternative hypothesis, H1). A Type 2 error would occur if the test fails to identify a defective product, allowing it to be shipped to customers, potentially causing harm or dissatisfaction.


#Suppose a pharmaceutical company is testing a new drug's effectiveness in treating a specific medical condition. They set up a clinical trial with the following hypotheses:
#Null Hypothesis (H0): The new drug is not effective in treating the condition.
#Alternative Hypothesis (H1): The new drug is effective in treating the condition.

#Let's consider airport security screening as an example:
#Null Hypothesis (H0): The passenger does not possess a dangerous item (e.g., weapon or explosive).
#Alternative Hypothesis (H1): The passenger possesses a dangerous item.


#Suppose you are working as a security screener at an airport. Your job is to determine whether passengers are carrying prohibited items in their luggage.
#Null Hypothesis (H0): The passenger's luggage does not contain prohibited items.
#Alternative Hypothesis (H1): The passenger's luggage contains prohibited items


#In airport security screening, passengers are screened for dangerous items, such as weapons or explosives. Let's define the hypotheses:
#Null Hypothesis (H0): The passenger does not possess any dangerous items.
#Alternative Hypothesis (H1): The passenger possesses dangerous items.

#
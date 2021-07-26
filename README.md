# Shap_Relational_Explanation

# Shap_Relational_Exp

In this project, we are explaining the predictions made by Machine Learning Model using SHAP (https://github.com/slundberg/shap). Here we have tried to represent SHAP relational explanations. The Yelp review data is used in this experiment and the data is available in the link (https://drive.google.com/drive/folders/1o-UmrtdLdYVTvUhWd75khzSqppP1upPq?usp=sharing). Here 2000 reviews (1000 positive reviews and 1000 negative reviews) have been used in the experiment. 

# Run the Code
 1. To get the visualized explanation outcomes in Relational_SHAP_EXPP notebooke run the command Run_program_all().

 2. In order to get explanation accuracy for the both word explanation, and relational explanations respectively in Relational_Shap_Accuracy notebook shap_accuracy() command needs to be run. 
 
 

# Demo Output 
Review 12 is Truly Predicted: Positive 

SHAP Relational Explanation: '131', '687', '593', '703', '150', '601', '212', '142'. Here '131', '687', '593', '703', '150', '601', '212', and  '142' are related reviews to 12 and have positive contribution to the prediction. 

SHAP Words Explanation: 'enjoy', 'drinks', 'show', 'champagne'. Here  are the word explanations

# Demo Explanation Accuracy
Word_Explanation_Average_Accuracy_Shap

0.278422782037238


Relational_Explanation_Average_Accuracy_Shap

0.5006702073849252



The explanation accuracy is computed separately for the word explanations, and relational explanations respectively. The manuall self-annotated (by human) process are considered to generate standard word explanations per query. The nerural network based embedding (Doc2Vec) is used to generate the standard relational explanations per query. Here, the standard explanations are considered as true explanations for the query being truly predicted as a specifice class. We have computed the percentage of SHAP relational model's explanations match with the standard explanations, and represent as explanation accuracy. In order to avoid biasness, we have kept the number of explanations  equal for the both standard explanations and SHAP relational model's explanations while computing explanation accuracy. Specifically, we have used first 5 explanations for the SHAP relational model's explanations based on the shap values, and standard explanations respectively. 

# Demo Visualized Output
# 1. Justification of the Prediction of the review query in terms of graph
The query review is currectly predicted as either positive or negative and from the relational graph as follows it is clear that, review 19 is truly predicted as negative as its related or connected review nodes either Samehotel or Sameuser relation are also negative reviews. 

![image](https://user-images.githubusercontent.com/25291998/126830625-d2cb30d0-09c5-46d3-9d96-ca2c24d1d649.png)


# 2. Sameuser Relation

Here, we have the relational graph where the query 19 predicted as negative review connected to a node with respect to Sameuser relation


![image](https://user-images.githubusercontent.com/25291998/126830658-c9abf7f3-8aac-4a86-adf1-4ad17dff9c21.png)


# 3. Samehotel Relation

Here, we have the relational graph where the query 4 predicted as positive review connected to the nodes with respect to Samehotel relation


![image](https://user-images.githubusercontent.com/25291998/126830694-2ab18a34-21a7-40f9-9898-675cec65f06e.png)

# 4. Word Explanations

Here are the words explanations by SHAP for the review 2 to be truly predicted as positive review:


![image](https://user-images.githubusercontent.com/25291998/126830752-6199cb90-56c3-404b-940f-75eb64e9bf6b.png)








# Packages need to be installed
Python=3.6

Gensim=3.8

Jupyter Notebook

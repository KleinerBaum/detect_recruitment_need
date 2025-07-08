# Recruitment Need Analysis Tool

## Setup

1. Install the dependencies:

```bash
pip install -r requirements.txt
```

2. Start the Streamlit interface:

```bash
streamlit run "Recruitment_Need_Analysis_Tool.py"
```

The **SKILLS** step suggests additional hard and soft skills via OpenAI and shows
them using your current job title for easier selection. It also displays a
dynamic salary chart that predicts the annual salary based on your selected
skills and other role information.

## Wizard Steps

The wizard collects data in the following order:

1. BASIC
2. COMPANY
3. DEPARTMENT
4. ROLE
5. TASKS
6. SKILLS
7. BENEFITS
8. TARGET_GROUP
9. INTERVIEW
10. SUMMARY

In the final **SUMMARY** step you can generate a job advertisement, an interview
guide, a Boolean search string and a draft contract. New buttons also let you
estimate the salary range and calculate the total annual compensation based on
selected benefits.

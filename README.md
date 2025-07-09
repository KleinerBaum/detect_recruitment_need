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

The **SKILLS** step suggests additional hard and soft skills via OpenAI and
presents them as selectable buttons. Chosen suggestions are stored in your
profile. A dynamic salary chart visualises the impact of your selected skills.

The **BENEFITS** step generates benefit suggestions based on the job title, the
company location and typical competitor offerings. Each suggestion appears as a
button that you can toggle to add it to your benefit list.

## Wizard Steps

The wizard collects data in the following order:

1. BASIC
2. COMPANY & DEPARTMENT – company and team details
3. ROLE & TASKS – responsibilities and key tasks
4. SKILLS
5. BENEFITS
6. TARGET_GROUP
7. INTERVIEW
8. SUMMARY

In the final **SUMMARY** step you can generate a job advertisement, an interview
guide, a Boolean search string and a draft contract. New buttons also let you
estimate the salary range and calculate the total annual compensation based on
selected benefits.

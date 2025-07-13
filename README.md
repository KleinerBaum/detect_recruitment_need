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
profile. Skills are grouped into language skills, key competencies and other
requirements for a cleaner layout. The interactive salary chart now lives in the
final **SUMMARY** step and shows how different factors influence the expected
annual salary.

The **BENEFITS** step generates benefit suggestions based on the job title, the
company location and typical competitor offerings. Each suggestion appears as a
button that you can toggle to add it to your benefit list. For some cities the
app proposes local perks such as club memberships or discounted facilities (for
example a "Fortuna Düsseldorf" membership when the location is Düsseldorf).

The **COMPANY & DEPARTMENT** step groups company info in a cleaner layout. The
*Team & Culture Context* now shows three bordered boxes. Each field provides a
**Generate Ideas** button above the input and AI suggestions appear as pill
buttons. Missing data remains highlighted in two columns below the extracted
values. After uploading a file a small success message with a 🔥 icon confirms
that the extraction finished. You can also paste a job ad URL to analyse the
posting directly.

Regex patterns for key information have been tightened and now support German
and English labels. Extracted values are validated by an LLM to increase
accuracy. Labels may now be prefixed with dashes or bullet characters.
If the patterns fail, a lightweight spaCy NER model guesses the company name
and city from the text.

## Wizard Steps

The wizard collects data in the following order:

1. BASIC
2. COMPANY & DEPARTMENT – company and team details
3. ROLE & TASKS – responsibilities and key tasks
4. SKILLS
5. BENEFITS
6. INTERVIEW – contact roles are grouped by Line Manager, HR and Finance. Application details and onboarding options are arranged side by side.
7. SUMMARY – includes Ideal Candidate Profile and Target Industries fields

In the final **SUMMARY** step you can generate a job advertisement, an interview
guide, a Boolean search string and a draft contract. New buttons also let you
estimate the salary range and calculate the total annual compensation based on
selected benefits.

## ESCO Integration

The tool includes helpers to query the [ESCO REST API](https://ec.europa.eu/esco/api) for
standardised occupations and skills. Environment variable `ESCO_API_BASE_URL` controls the
target endpoint. Occupation lookup and ESCO skill suggestions are available directly in the **SKILLS** step.

The helper uses the ``hasEssentialSkill`` relation to fetch essential skills for
an occupation from the ESCO API.

## Vector Search

Missing fields can be filled via Retrieval-Augmented Generation. Relevant job ads
and CV snippets are retrieved from the pre-trained "vacalyser" vector store and
passed to the LLM. Set ``VACALYSER_VECTOR_STORE`` to override the default store
ID.

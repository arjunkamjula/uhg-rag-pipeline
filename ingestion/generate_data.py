"""
ingestion/generate_data.py

Synthetic healthcare data generator for the UHG RAG pipeline.

Generates:
  - 5,000 member records       -> PostgreSQL members table
  - 50,000 claim records       -> PostgreSQL claims table
  - 8,000 prior auth records   -> PostgreSQL prior_auth table
  - 2,000 EOB PDFs             -> data/raw/eob/
  - 1,500 clinical note PDFs   -> data/raw/clinical_notes/
  - 800  prior auth PDFs       -> data/raw/prior_auth/
  - 600  denial letter PDFs    -> data/raw/denial_letters/
  - 300  appeal letter PDFs    -> data/raw/appeal_letters/
  - 1,000 FHIR R4 JSON records -> data/raw/fhir/
  - 500  case manager TXT notes-> data/raw/case_notes/
  - 3   CSV exports            -> data/raw/csv_exports/

Run:
    python ingestion/generate_data.py
"""

import csv
import json
import os
import random
from datetime import datetime, timedelta
from pathlib import Path

import fitz
from faker import Faker
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

fake = Faker()
random.seed(42)
Faker.seed(42)

RAW_DIR = Path("data/raw")
for sub in ["eob", "clinical_notes", "prior_auth", "denial_letters",
            "appeal_letters", "fhir", "csv_exports", "case_notes"]:
    (RAW_DIR / sub).mkdir(parents=True, exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)

DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://uhg_user:uhg_pass@localhost:5433/uhg_claims"
)

CPT_CODES = {
    "27447": "Total knee arthroplasty",
    "27130": "Total hip arthroplasty",
    "29881": "Knee arthroscopy with meniscectomy",
    "43239": "Upper GI endoscopy with biopsy",
    "45378": "Colonoscopy diagnostic",
    "70553": "MRI brain with contrast",
    "71250": "CT thorax without contrast",
    "93306": "Echocardiography transthoracic",
    "99213": "Office visit established patient moderate",
    "99214": "Office visit established patient complex",
    "99232": "Subsequent hospital care",
    "90837": "Psychotherapy 60 minutes",
    "20610": "Arthrocentesis major joint",
    "64483": "Injection epidural lumbar",
    "27035": "Denervation hip joint",
}

ICD10_CODES = {
    "M17.11": "Primary osteoarthritis right knee",
    "M16.11": "Primary osteoarthritis right hip",
    "M23.200": "Derangement of unspecified meniscus",
    "K21.0":  "GERD with esophagitis",
    "Z12.11": "Encounter screening colon cancer",
    "G35":    "Multiple sclerosis",
    "J18.9":  "Pneumonia unspecified organism",
    "I25.10": "Atherosclerotic heart disease native vessel",
    "F32.1":  "Major depressive disorder single moderate",
    "M54.5":  "Low back pain",
    "M54.4":  "Lumbago with sciatica",
    "E11.9":  "Type 2 diabetes without complications",
    "I10":    "Essential hypertension",
    "J44.1":  "COPD with acute exacerbation",
    "N39.0":  "Urinary tract infection",
}

PLAN_TYPES = [
    "UHC-PPO-Gold", "UHC-PPO-Silver", "UHC-HMO-Standard",
    "UHC-HDHP-Bronze", "UHC-Medicare-Advantage", "UHC-Medicaid-Managed",
]

DENIAL_REASONS = {
    "CO-4":   "Service inconsistent with payer coverage determination",
    "CO-11":  "Diagnosis inconsistent with procedure",
    "CO-16":  "Claim lacks information for adjudication",
    "CO-29":  "Time limit for filing has expired",
    "CO-50":  "Not medically necessary",
    "CO-97":  "Payment included in allowance for another service",
    "PR-1":   "Deductible amount",
    "PR-2":   "Coinsurance amount",
    "OA-109": "Claim not covered by this payer",
}

PROVIDERS = [
    ("Dr. Sarah Mitchell", "NPI-1234567890", "Orthopedics"),
    ("Dr. James Okonkwo",  "NPI-2345678901", "Cardiology"),
    ("Dr. Priya Sharma",   "NPI-3456789012", "Gastroenterology"),
    ("Dr. Michael Torres", "NPI-4567890123", "Neurology"),
    ("Dr. Linda Vasquez",  "NPI-5678901234", "Primary Care"),
    ("Dr. Robert Chen",    "NPI-6789012345", "Psychiatry"),
    ("Dr. Angela Brooks",  "NPI-7890123456", "Radiology"),
]

CASE_MANAGERS = [
    "CM-Jennifer Walsh", "CM-David Osei",
    "CM-Maria Gutierrez", "CM-Thomas Park",
    "CM-Sandra Okafor", "CM-Kevin Huang",
]

MEDICATIONS = [
    "Metformin 500mg", "Lisinopril 10mg", "Atorvastatin 40mg",
    "Amlodipine 5mg", "Omeprazole 20mg", "Levothyroxine 50mcg",
    "Metoprolol 25mg", "Gabapentin 300mg", "Sertraline 50mg",
    "Ibuprofen 600mg",
]

ALLERGIES = [
    "Penicillin — rash", "Sulfa — anaphylaxis",
    "Codeine — nausea", "Aspirin — GI bleed",
    "Latex — contact dermatitis", "NKDA",
]

STATES = ["TX", "CA", "FL", "NY", "IL", "OH", "PA", "GA", "NC", "MI"]


def new_member_id():
    return f"M-{random.randint(10000, 99999)}"


def new_claim_id():
    return f"CLM-{random.randint(10000, 99999)}"


def new_auth_id():
    return f"PA-{random.randint(1000, 9999)}"


def rand_date(start_year=2021, end_year=2024):
    start = datetime(start_year, 1, 1)
    delta = (datetime(end_year, 12, 31) - start).days
    return start + timedelta(days=random.randint(0, delta))


def rand_cpt():
    return random.choice(list(CPT_CODES.items()))


def rand_icd():
    return random.choice(list(ICD10_CODES.items()))


def rand_provider():
    return random.choice(PROVIDERS)


def billed_amount(cpt_code):
    ranges = {
        "27447": (18000, 35000), "27130": (16000, 30000),
        "29881": (4000, 8000),   "43239": (1500, 3000),
        "45378": (1200, 2500),   "70553": (800, 1800),
        "71250": (600, 1400),    "93306": (500, 1200),
        "99213": (150, 300),     "99214": (200, 450),
        "99232": (250, 500),     "90837": (180, 350),
        "20610": (300, 600),     "64483": (800, 1600),
        "27035": (1200, 2400),
    }
    lo, hi = ranges.get(cpt_code, (200, 5000))
    return round(random.uniform(lo, hi), 2)


def write_pdf(lines: list, path: Path):
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    y = 60
    for line in lines:
        if line.startswith("##TITLE##"):
            txt = line.replace("##TITLE##", "").strip()
            page.insert_text((50, y), txt, fontsize=14, fontname="helv")
            y += 24
        elif line.startswith("##HEAD##"):
            txt = line.replace("##HEAD##", "").strip()
            page.insert_text((50, y), txt, fontsize=11, fontname="helv")
            y += 18
        elif line == "##HR##":
            page.draw_line((50, y), (562, y), width=0.5)
            y += 10
        elif line == "":
            y += 8
        else:
            words = line.split()
            cur = ""
            for w in words:
                test = cur + (" " if cur else "") + w
                if len(test) > 95:
                    page.insert_text((50, y), cur, fontsize=9, fontname="helv")
                    y += 13
                    cur = w
                    if y > 745:
                        page = doc.new_page(width=612, height=792)
                        y = 60
                else:
                    cur = test
            if cur:
                page.insert_text((50, y), cur, fontsize=9, fontname="helv")
                y += 13
        if y > 745:
            page = doc.new_page(width=612, height=792)
            y = 60
    doc.save(str(path))
    doc.close()


def eob_lines(member, claim):
    cpt_desc = CPT_CODES[claim["cpt_code"]]
    icd_desc = ICD10_CODES[claim["icd_code"]]
    provider = rand_provider()
    billed   = claim["billed_amount"]
    allowed  = round(billed * random.uniform(0.55, 0.75), 2)
    ded      = round(random.uniform(0, min(500, allowed * 0.2)), 2)
    coins    = round((allowed - ded) * random.uniform(0.1, 0.2), 2)
    paid     = round(allowed - ded - coins, 2)

    lines = [
        "##TITLE##EXPLANATION OF BENEFITS",
        "##HEAD##UnitedHealthcare — Member Services",
        "##HR##", "",
        f"Member Name:        {member['name']}",
        f"Member ID:          {member['member_id']}",
        f"Plan:               {member['plan_type']}",
        f"Group Number:       GRP-{random.randint(10000, 99999)}",
        f"Claim Number:       {claim['claim_id']}",
        f"Date of Service:    {claim['service_date']}",
        f"Processed Date:     {claim['processed_date']}",
        "", "##HEAD##SERVICE DETAIL", "##HR##", "",
        f"Procedure Code:     {claim['cpt_code']}",
        f"Procedure:          {cpt_desc}",
        f"Diagnosis Code:     {claim['icd_code']}",
        f"Diagnosis:          {icd_desc}",
        f"Rendering Provider: {provider[0]}",
        f"Provider NPI:       {provider[1]}",
        f"Specialty:          {provider[2]}",
        "", "##HEAD##PAYMENT SUMMARY", "##HR##", "",
        f"Amount Billed:      ${billed:,.2f}",
        f"Allowed Amount:     ${allowed:,.2f}",
        f"Deductible Applied: ${ded:,.2f}",
        f"Coinsurance (20%):  ${coins:,.2f}",
        f"Plan Paid:          ${paid:,.2f}",
        f"Member Responsibility: ${ded + coins:,.2f}",
        f"Claim Status:       {claim['status'].upper()}",
        "",
    ]

    if claim["status"] == "denied":
        dc, dd = random.choice(list(DENIAL_REASONS.items()))
        lines += [
            "##HEAD##DENIAL INFORMATION", "##HR##", "",
            f"Denial Reason Code: {dc}",
            f"Denial Reason:      {dd}",
            "",
            "This claim has been denied. You have the right to appeal within 180 days.",
            "Contact UHC Member Services at 1-800-328-5979.",
            "",
        ]

    lines += [
        "##HEAD##IMPORTANT NOTICES", "##HR##", "",
        "This is not a bill. Retain this EOB for your records.",
        f"Document ID: EOB-{claim['claim_id']}-{member['member_id']}",
        f"Generated:   {datetime.now().strftime('%Y-%m-%d')}",
    ]
    return lines


def clinical_note_lines(member, claim):
    cpt_desc = CPT_CODES[claim["cpt_code"]]
    icd_desc = ICD10_CODES[claim["icd_code"]]
    provider = rand_provider()
    age      = random.randint(35, 78)
    bmi      = round(random.uniform(21, 38), 1)
    bp       = f"{random.randint(110, 145)}/{random.randint(70, 95)}"
    duration = random.choice(["6 months", "12 months", "18 months", "2 years", "3 years"])
    cons_tx  = random.choice([
        "physical therapy, NSAIDs, and corticosteroid injections",
        "physical therapy, activity modification, and oral analgesics",
        "rest, ice, compression, and anti-inflammatory medications",
        "chiropractic care, physical therapy, and analgesics",
    ])
    imaging  = random.choice([
        "X-ray confirms grade III-IV joint space narrowing",
        "MRI demonstrates full-thickness cartilage loss and subchondral edema",
        "CT imaging reveals significant degenerative changes",
        "X-ray and MRI findings consistent with advanced degenerative joint disease",
    ])
    impair   = random.choice([
        "Patient reports inability to ambulate more than one block without severe pain.",
        "Activities of daily living significantly impaired. Unable to climb stairs.",
        "Patient experiences rest pain and nocturnal pain affecting sleep quality.",
        "Severe functional limitation affecting work capacity and quality of life.",
    ])

    return [
        "##TITLE##CLINICAL NOTE / DISCHARGE SUMMARY",
        f"##HEAD##Attending: {provider[0]}   NPI: {provider[1]}   Specialty: {provider[2]}",
        "##HR##", "",
        f"Patient:            {member['name']}",
        f"Member ID:          {member['member_id']}",
        f"DOB:                {member['dob']}",
        f"Age:                {age}",
        f"Date of Service:    {claim['service_date']}",
        f"Claim Reference:    {claim['claim_id']}",
        f"Insurance Plan:     {member['plan_type']}",
        "", "##HEAD##CHIEF COMPLAINT", "##HR##", "",
        f"Patient presents with {icd_desc.lower()} with functional impairment.",
        impair,
        "", "##HEAD##HISTORY OF PRESENT ILLNESS", "##HR##", "",
        f"Patient is a {age}-year-old with a {duration} history of worsening symptoms",
        f"consistent with {icd_desc} (ICD-10: {claim['icd_code']}). Managed conservatively",
        f"with {cons_tx} without adequate relief. {imaging}.",
        "Conservative management exhausted. Surgical intervention now indicated.",
        "", "##HEAD##PHYSICAL EXAMINATION", "##HR##", "",
        f"Vitals: BP {bp} mmHg, BMI {bmi} kg/m2",
        "General: Alert and oriented x3, no acute distress at rest.",
        "Musculoskeletal: Tenderness to palpation. ROM limited. Crepitus noted.",
        "Neurovascular: Distal pulses intact. Sensation intact distally.",
        "", "##HEAD##ASSESSMENT AND PLAN", "##HR##", "",
        f"Primary Diagnosis: {icd_desc} ({claim['icd_code']})",
        f"Recommended Procedure: {cpt_desc} (CPT: {claim['cpt_code']})",
        "",
        f"Patient is medically appropriate for {cpt_desc}. No active contraindications.",
        "Risks and benefits discussed. Informed consent obtained.",
        "", "##HEAD##CLINICAL CRITERIA FOR AUTHORIZATION", "##HR##", "",
        f"1. Diagnosis of {icd_desc} confirmed by imaging and clinical exam.",
        f"2. Duration of symptoms: {duration} — exceeds minimum 3-month threshold.",
        f"3. Conservative treatment trial: {cons_tx} — failed.",
        "4. Functional impairment documented affecting ADLs.",
        "5. Patient medically fit. No active contraindications identified.",
        "",
        f"Electronically signed: {provider[0]}   Date: {claim['service_date']}",
    ]


def prior_auth_lines(member, auth):
    cpt_desc = CPT_CODES.get(auth["procedure_code"], "Requested procedure")
    icd_code, icd_desc = rand_icd()
    provider = rand_provider()
    decision = auth["decision"]

    lines = [
        "##TITLE##PRIOR AUTHORIZATION DECISION",
        "##HEAD##UnitedHealthcare Clinical Services",
        "##HR##", "",
        f"Authorization ID:    {auth['auth_id']}",
        f"Member Name:         {member['name']}",
        f"Member ID:           {member['member_id']}",
        f"Plan:                {member['plan_type']}",
        f"Requesting Provider: {provider[0]}  (NPI: {provider[1]})",
        f"Date of Request:     {auth['request_date']}",
        f"Decision Date:       {auth['decision_date']}",
        "", "##HEAD##REQUESTED SERVICE", "##HR##", "",
        f"Procedure Code:  {auth['procedure_code']}",
        f"Procedure:       {cpt_desc}",
        f"Diagnosis Code:  {icd_code}",
        f"Diagnosis:       {icd_desc}",
        "", "##HEAD##AUTHORIZATION DECISION", "##HR##", "",
        f"DECISION: {decision.upper()}",
        "",
    ]

    if decision == "approved":
        valid_to = (
            datetime.strptime(auth["decision_date"], "%Y-%m-%d") + timedelta(days=90)
        ).strftime("%Y-%m-%d")
        lines += [
            f"Service APPROVED. Valid from {auth['decision_date']} to {valid_to}.",
            "Criteria met: diagnosis confirmed, conservative treatment failed,",
            "procedure medically necessary, provider credentialed.",
            f"Authorization Number: {auth['auth_id']} — include on all claims.",
        ]
    elif decision == "denied":
        dc, dd = random.choice(list(DENIAL_REASONS.items()))
        lines += [
            f"Service DENIED. Denial Code: {dc}",
            f"Reason: {dd}",
            "",
            "Submitted documentation does not meet UHG clinical criteria.",
            "Appeal rights: submit written appeal within 180 days.",
            "UHC Appeals, P.O. Box 30432, Salt Lake City, UT 84130",
        ]
    else:
        lines += [
            "Request PENDING. Additional clinical information required.",
            "Expected decision: within 3 business days of receipt.",
        ]

    lines += [
        "", "##HEAD##NOTICE", "##HR##", "",
        "Authorization is not a guarantee of payment.",
        f"Document ID: AUTH-{auth['auth_id']}-{member['member_id']}",
    ]
    return lines


def denial_letter_lines(member, claim):
    cpt_desc = CPT_CODES[claim["cpt_code"]]
    icd_desc = ICD10_CODES[claim["icd_code"]]
    dc, dd   = random.choice(list(DENIAL_REASONS.items()))
    provider = rand_provider()

    return [
        "##TITLE##NOTICE OF CLAIM DENIAL",
        "##HEAD##UnitedHealthcare Member Services",
        "##HR##", "",
        f"Date:               {claim['processed_date']}",
        f"Member Name:        {member['name']}",
        f"Member ID:          {member['member_id']}",
        f"Claim Number:       {claim['claim_id']}",
        f"Date of Service:    {claim['service_date']}",
        f"Provider:           {provider[0]}",
        "", "##HEAD##DENIED SERVICE", "##HR##", "",
        f"Procedure Code:     {claim['cpt_code']}",
        f"Procedure:          {cpt_desc}",
        f"Diagnosis:          {icd_desc} ({claim['icd_code']})",
        f"Amount Billed:      ${claim['billed_amount']:,.2f}",
        f"Amount Denied:      ${claim['billed_amount']:,.2f}",
        "", "##HEAD##REASON FOR DENIAL", "##HR##", "",
        f"Denial Code:        {dc}",
        f"Reason:             {dd}",
        "",
        f"Clinical documentation does not support medical necessity for {cpt_desc}",
        f"under applicable UHG clinical policy guidelines for plan {member['plan_type']}.",
        "", "##HEAD##YOUR APPEAL RIGHTS", "##HR##", "",
        "First-level appeal must be submitted within 180 calendar days.",
        "Submit to: UHC Appeals, P.O. Box 30432, Salt Lake City, UT 84130",
        "Phone: 1-866-892-7047   Fax: 1-801-938-2100",
        "",
        f"Document ID: DENIAL-{claim['claim_id']}-{member['member_id']}",
    ]


def appeal_letter_lines(member, claim):
    cpt_desc = CPT_CODES[claim["cpt_code"]]
    icd_desc = ICD10_CODES[claim["icd_code"]]
    provider = rand_provider()
    outcome  = random.choice(["overturned", "upheld"])
    dec_date = (
        datetime.strptime(claim["processed_date"], "%Y-%m-%d") +
        timedelta(days=random.randint(14, 45))
    ).strftime("%Y-%m-%d")

    lines = [
        "##TITLE##APPEAL DECISION NOTICE",
        "##HEAD##UnitedHealthcare — Appeals and Grievances",
        "##HR##", "",
        f"Appeal Reference:   APL-{claim['claim_id']}",
        f"Member Name:        {member['name']}",
        f"Member ID:          {member['member_id']}",
        f"Original Claim:     {claim['claim_id']}",
        f"Appeal Date:        {claim['processed_date']}",
        f"Decision Date:      {dec_date}",
        "", "##HEAD##APPEAL SUMMARY", "##HR##", "",
        f"Procedure:          {cpt_desc} (CPT: {claim['cpt_code']})",
        f"Diagnosis:          {icd_desc} ({claim['icd_code']})",
        f"Rendering Provider: {provider[0]}",
        f"Appeal Outcome:     {outcome.upper()}",
        "", "##HEAD##DECISION RATIONALE", "##HR##", "",
    ]

    if outcome == "overturned":
        lines += [
            "After review of additional clinical documentation, the original denial",
            "is OVERTURNED. Clinical criteria now satisfied.",
            "Claim will be reprocessed. Payment issued within 30 business days.",
        ]
    else:
        lines += [
            "After thorough review the original denial is UPHELD.",
            "Submitted documentation does not meet required clinical criteria.",
            "External review by an Independent Review Organization is available.",
        ]

    lines += [
        "",
        "Reviewed by: UHC Clinical Appeals Team — Board Certified Reviewers",
        f"Document ID: APL-{claim['claim_id']}-{member['member_id']}",
    ]
    return lines


def generate_fhir_record(member, claims_for_member):
    conditions = list({
        c["icd_code"]: {
            "code":    c["icd_code"],
            "display": ICD10_CODES[c["icd_code"]]
        }
        for c in claims_for_member[:5]
    }.values())

    encounters = [
        {
            "encounter_id": f"ENC-{c['claim_id']}",
            "date":         c["service_date"],
            "type":         CPT_CODES[c["cpt_code"]],
            "cpt_code":     c["cpt_code"],
            "status":       c["status"],
            "provider":     rand_provider()[0],
        }
        for c in claims_for_member[:8]
    ]

    return {
        "resourceType": "Bundle",
        "id": f"bundle-{member['member_id']}",
        "type": "collection",
        "timestamp": datetime.now().isoformat(),
        "entry": [
            {
                "resourceType": "Patient",
                "id": member["member_id"],
                "name": member["name"],
                "birthDate": member["dob"],
                "gender": random.choice(["male", "female"]),
                "address": {
                    "state": member["state"],
                    "postalCode": fake.postcode(),
                },
                "insurance": {
                    "plan": member["plan_type"],
                    "memberId": member["member_id"],
                    "groupNumber": f"GRP-{random.randint(10000, 99999)}",
                    "enrollmentDate": member["enrollment_date"],
                    "status": member["status"],
                },
            },
            {
                "resourceType": "ConditionList",
                "conditions": conditions,
            },
            {
                "resourceType": "MedicationList",
                "medications": random.sample(MEDICATIONS, k=random.randint(2, 5)),
            },
            {
                "resourceType": "AllergyList",
                "allergies": [random.choice(ALLERGIES)],
            },
            {
                "resourceType": "EncounterList",
                "encounters": encounters,
            },
            {
                "resourceType": "LabResults",
                "results": [
                    {"test": "HbA1c",     "value": round(random.uniform(5.2, 11.0), 1), "unit": "%"},
                    {"test": "LDL",       "value": random.randint(70, 210),              "unit": "mg/dL"},
                    {"test": "Creatinine","value": round(random.uniform(0.6, 2.1), 2),   "unit": "mg/dL"},
                    {"test": "eGFR",      "value": random.randint(30, 110),              "unit": "mL/min"},
                ],
            },
        ],
    }


def generate_case_note(member, claim):
    cm       = random.choice(CASE_MANAGERS)
    cpt_desc = CPT_CODES[claim["cpt_code"]]
    icd_desc = ICD10_CODES[claim["icd_code"]]
    date     = rand_date().strftime("%Y-%m-%d")

    contact_type = random.choice([
        "Outbound phone call", "Inbound phone call",
        "Secure message", "Care coordination note",
    ])
    outcome = random.choice([
        f"Member confirmed awareness of claim status for {claim['claim_id']}.",
        f"Educated member on appeal rights for denied claim {claim['claim_id']}.",
        f"Coordinated with provider {rand_provider()[0]} regarding prior auth.",
        "Member expressed satisfaction with resolution. Case closed.",
        f"Confirmed member scheduled follow-up appointment for {cpt_desc}.",
    ])

    lines = [
        "=" * 60,
        "CASE MANAGER NOTE — CONFIDENTIAL",
        "=" * 60,
        f"Date:           {date}",
        f"Case Manager:   {cm}",
        f"Member ID:      {member['member_id']}",
        f"Member Name:    {member['name']}",
        f"Plan:           {member['plan_type']}",
        f"Contact Type:   {contact_type}",
        f"Claim Ref:      {claim['claim_id']}",
        "",
        "SUMMARY:",
        f"Contact initiated regarding {icd_desc} claim (CPT: {claim['cpt_code']},",
        f"ICD-10: {claim['icd_code']}). Claim current status: {claim['status'].upper()}.",
        "",
        "NOTES:",
        outcome,
        "",
        "FOLLOW-UP:",
        random.choice([
            "No follow-up required. Case closed.",
            "Follow-up call scheduled in 7 days.",
            "Awaiting provider documentation. Follow-up in 3 days.",
        ]),
        "",
        f"Next Review Date: {(datetime.strptime(date, '%Y-%m-%d') + timedelta(days=random.randint(7, 30))).strftime('%Y-%m-%d')}",
        f"Document ID: CN-{claim['claim_id']}-{member['member_id']}-{date}",
        "=" * 60,
    ]
    return "\n".join(lines)


def populate_database(members, claims, auths):
    print("  Connecting to PostgreSQL...")
    engine = create_engine(DB_URL)

    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS prior_auth CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS claims CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS members CASCADE"))

        conn.execute(text("""
            CREATE TABLE members (
                member_id       VARCHAR(20) PRIMARY KEY,
                name            VARCHAR(100),
                dob             DATE,
                plan_type       VARCHAR(50),
                state           VARCHAR(2),
                enrollment_date DATE,
                status          VARCHAR(20)
            )
        """))
        conn.execute(text("""
            CREATE TABLE claims (
                claim_id        VARCHAR(20) PRIMARY KEY,
                member_id       VARCHAR(20) REFERENCES members(member_id),
                cpt_code        VARCHAR(10),
                icd_code        VARCHAR(10),
                billed_amount   NUMERIC(10,2),
                allowed_amount  NUMERIC(10,2),
                plan_paid       NUMERIC(10,2),
                status          VARCHAR(20),
                service_date    DATE,
                processed_date  DATE,
                provider_npi    VARCHAR(20),
                denial_code     VARCHAR(10)
            )
        """))
        conn.execute(text("""
            CREATE TABLE prior_auth (
                auth_id         VARCHAR(20) PRIMARY KEY,
                member_id       VARCHAR(20) REFERENCES members(member_id),
                claim_id        VARCHAR(20),
                procedure_code  VARCHAR(10),
                decision        VARCHAR(20),
                request_date    DATE,
                decision_date   DATE
            )
        """))

        print(f"  Inserting {len(members):,} members...")
        for m in members:
            conn.execute(text("""
                INSERT INTO members VALUES
                (:member_id,:name,:dob,:plan_type,:state,:enrollment_date,:status)
            """), {k: v for k, v in m.items()
                   if k in ["member_id", "name", "dob", "plan_type",
                             "state", "enrollment_date", "status"]})

        print(f"  Inserting {len(claims):,} claims...")
        for c in claims:
            conn.execute(text("""
                INSERT INTO claims VALUES
                (:claim_id,:member_id,:cpt_code,:icd_code,:billed_amount,
                 :allowed_amount,:plan_paid,:status,:service_date,
                 :processed_date,:provider_npi,:denial_code)
            """), c)

        print(f"  Inserting {len(auths):,} prior auth records...")
        for a in auths:
            conn.execute(text("""
                INSERT INTO prior_auth VALUES
                (:auth_id,:member_id,:claim_id,:procedure_code,
                 :decision,:request_date,:decision_date)
            """), a)

    print("  Database population complete.")


def write_csv_exports(members, claims, auths):
    fields = ["claim_id", "member_id", "cpt_code", "icd_code", "billed_amount",
              "allowed_amount", "plan_paid", "status", "service_date",
              "processed_date", "provider_npi", "denial_code"]

    with open(RAW_DIR / "csv_exports" / "claims_export.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for c in claims:
            w.writerow({k: c.get(k, "") for k in fields})

    with open(RAW_DIR / "csv_exports" / "eligibility_export.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "member_id", "name", "dob", "plan_type", "state",
            "enrollment_date", "status"
        ])
        w.writeheader()
        for m in members:
            w.writerow({k: m.get(k, "") for k in
                        ["member_id", "name", "dob", "plan_type",
                         "state", "enrollment_date", "status"]})

    with open(RAW_DIR / "csv_exports" / "prior_auth_export.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "auth_id", "member_id", "claim_id", "procedure_code",
            "decision", "request_date", "decision_date"
        ])
        w.writeheader()
        for a in auths:
            w.writerow(a)

    print(f"  CSV exports written to {RAW_DIR / 'csv_exports'}")


def main():
    print("=" * 60)
    print("UHG RAG Pipeline — Synthetic Data Generator")
    print("=" * 60)

    print("\n[1/8] Generating 5,000 members...")
    members    = []
    member_ids = set()
    while len(members) < 5000:
        mid = new_member_id()
        if mid in member_ids:
            continue
        member_ids.add(mid)
        members.append({
            "member_id":       mid,
            "name":            fake.name(),
            "dob":             rand_date(1945, 1990).strftime("%Y-%m-%d"),
            "plan_type":       random.choice(PLAN_TYPES),
            "state":           random.choice(STATES),
            "enrollment_date": rand_date(2018, 2022).strftime("%Y-%m-%d"),
            "status":          random.choices(["active", "inactive"], weights=[85, 15])[0],
        })
    member_lookup = {m["member_id"]: m for m in members}

    print("[2/8] Generating 50,000 claims...")
    claims    = []
    claim_ids = set()
    while len(claims) < 50000:
        cid = new_claim_id()
        if cid in claim_ids:
            continue
        claim_ids.add(cid)
        member      = random.choice(members)
        cpt_code, _ = rand_cpt()
        icd_code, _ = rand_icd()
        billed      = billed_amount(cpt_code)
        allowed     = round(billed * random.uniform(0.55, 0.75), 2)
        plan_paid   = round(allowed * random.uniform(0.70, 0.90), 2)
        status      = random.choices(
            ["approved", "denied", "pending"], weights=[65, 25, 10]
        )[0]
        svc = rand_date(2021, 2024)
        claims.append({
            "claim_id":       cid,
            "member_id":      member["member_id"],
            "cpt_code":       cpt_code,
            "icd_code":       icd_code,
            "billed_amount":  billed,
            "allowed_amount": allowed,
            "plan_paid":      plan_paid,
            "status":         status,
            "service_date":   svc.strftime("%Y-%m-%d"),
            "processed_date": (svc + timedelta(days=random.randint(3, 30))).strftime("%Y-%m-%d"),
            "provider_npi":   rand_provider()[1],
            "denial_code":    random.choice(list(DENIAL_REASONS.keys())) if status == "denied" else None,
        })

    claims_by_member = {}
    for c in claims:
        claims_by_member.setdefault(c["member_id"], []).append(c)

    print("[3/8] Generating 8,000 prior auth records...")
    auths    = []
    auth_ids = set()
    while len(auths) < 8000:
        aid = new_auth_id()
        if aid in auth_ids:
            continue
        auth_ids.add(aid)
        member      = random.choice(members)
        cpt_code, _ = rand_cpt()
        req         = rand_date(2021, 2024)
        auths.append({
            "auth_id":        aid,
            "member_id":      member["member_id"],
            "claim_id":       random.choice(claims)["claim_id"],
            "procedure_code": cpt_code,
            "decision":       random.choices(
                ["approved", "denied", "pending"], weights=[60, 30, 10]
            )[0],
            "request_date":   req.strftime("%Y-%m-%d"),
            "decision_date":  (req + timedelta(days=random.randint(1, 5))).strftime("%Y-%m-%d"),
        })

    print("[4/8] Populating PostgreSQL...")
    try:
        populate_database(members, claims, auths)
    except Exception as e:
        print(f"  WARNING: DB skipped ({e})")
        print("  Run docker-compose up -d first.")

    print("[5/8] Writing CSV exports...")
    write_csv_exports(members, claims, auths)

    denied_claims = [c for c in claims if c["status"] == "denied"]
    counts = {
        "eob": 0, "clinical_note": 0, "prior_auth": 0,
        "denial": 0, "appeal": 0, "fhir": 0, "case_note": 0,
    }

    print("[6/8] Generating PDFs...")
    for claim in random.sample(claims, 2000):
        m = member_lookup[claim["member_id"]]
        p = RAW_DIR / "eob" / f"EOB_{claim['claim_id']}_{m['member_id']}_{claim['service_date']}.pdf"
        write_pdf(eob_lines(m, claim), p)
        counts["eob"] += 1

    for claim in random.sample(claims, 1500):
        m = member_lookup[claim["member_id"]]
        p = RAW_DIR / "clinical_notes" / f"ClinicalNote_{claim['claim_id']}_{m['member_id']}_{claim['service_date']}.pdf"
        write_pdf(clinical_note_lines(m, claim), p)
        counts["clinical_note"] += 1

    for auth in random.sample(auths, 800):
        m = member_lookup[auth["member_id"]]
        p = RAW_DIR / "prior_auth" / f"PriorAuth_{auth['auth_id']}_{m['member_id']}_{auth['decision_date']}.pdf"
        write_pdf(prior_auth_lines(m, auth), p)
        counts["prior_auth"] += 1

    for claim in random.sample(denied_claims, min(600, len(denied_claims))):
        m = member_lookup[claim["member_id"]]
        p = RAW_DIR / "denial_letters" / f"DenialLetter_{claim['claim_id']}_{m['member_id']}_{claim['processed_date']}.pdf"
        write_pdf(denial_letter_lines(m, claim), p)
        counts["denial"] += 1

    for claim in random.sample(denied_claims, min(300, len(denied_claims))):
        m = member_lookup[claim["member_id"]]
        p = RAW_DIR / "appeal_letters" / f"AppealLetter_{claim['claim_id']}_{m['member_id']}_{claim['processed_date']}.pdf"
        write_pdf(appeal_letter_lines(m, claim), p)
        counts["appeal"] += 1

    print("[7/8] Generating FHIR JSON + TXT case notes...")
    for member in random.sample(members, 1000):
        mc = claims_by_member.get(member["member_id"], [])
        if not mc:
            continue
        record = generate_fhir_record(member, mc)
        p = RAW_DIR / "fhir" / f"FHIR_{member['member_id']}.json"
        with open(p, "w") as f:
            json.dump(record, f, indent=2)
        counts["fhir"] += 1

    for claim in random.sample(claims, 500):
        m    = member_lookup[claim["member_id"]]
        note = generate_case_note(m, claim)
        p    = RAW_DIR / "case_notes" / f"CaseNote_{claim['claim_id']}_{m['member_id']}.txt"
        p.write_text(note, encoding="utf-8")
        counts["case_note"] += 1

    print("[8/8] Writing manifest...")
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "structured_db": {
            "members":     len(members),
            "claims":      len(claims),
            "prior_auths": len(auths),
        },
        "unstructured_files": counts,
        "total_files": sum(counts.values()),
        "formats": ["PDF", "JSON (FHIR R4)", "CSV", "TXT"],
    }
    with open("data/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Structured DB : {len(members):,} members | {len(claims):,} claims | {len(auths):,} auths")
    print(f"  PDFs          : {counts['eob']} EOB | {counts['clinical_note']} clinical | "
          f"{counts['prior_auth']} auth | {counts['denial']} denial | {counts['appeal']} appeal")
    print(f"  JSON (FHIR)   : {counts['fhir']} patient records")
    print(f"  TXT           : {counts['case_note']} case manager notes")
    print(f"  CSV           : 3 export files")
    print(f"  Total files   : {sum(counts.values()) + 3:,}")
    print("=" * 60)


if __name__ == "__main__":
    main()

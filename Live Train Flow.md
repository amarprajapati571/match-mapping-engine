알겠어. 아래는 **AI팀(Python)** \+ **스포츠 플랫폼팀(백엔드+어드민 UI)** 에게 **한 번에 내려도 되는 “종합 Task Flow (MVP Ver 4.0 \+ 필수 3개 추가 반영)”** 최종 영어 문서야. 그대로 복붙해서 보내면 돼.

---

### **Final Specification (MVP): AI-Assisted Match Mapping System (Ver. 4.0)**

## **1\) Core Concept**

We will use an **Active Learning** workflow:

**AI proposes Top-5 candidates → Humans verify → Feedback is stored → Retraining improves automation**  
Goal: achieve **90%+ live accuracy for “match-exists” cases**, while safely handling “no match” cases.

---

## **2\) End-to-End Task Flow & Responsibilities**

### **Step 1 — Data Supply (Platform → AI)**

**Platform Team**

* Provide APIs (or DB access) to fetch raw match data from:  
  * **OddsPortal (OP)** matches (source list)  
  * **Bet365 (B365)** matches (candidate pool)  
* Provide enough fields for matching:  
  * `match_id`, `sport`, `home`, `away`, `kickoff_time`, (+ optional league)  
  * `category_gender`: `M | W | Mixed | Unknown`  
    `category_age`: `Senior U21 | U| U23 | 19 | U17 | ... | Unknown`  
    `team_level`: `First | Reserves | B | II | Academy | Unknown`

**AI Team **

* Periodically pull “new/unmapped” OP matches.  
* Mark them as `NEW` for processing (in AI or Platform status store).

---

### **Step 2 — AI Inference & Candidate Generation (AI Team)**

For each OP match:

1. **Pre-filter candidate pool**  
* Candidate pool must be restricted to:  
  * **Same sport**  
  * **Kickoff time within ±30 minutes**  
2. **Candidate retrieval \+ reranking**  
* Use **SBERT** to retrieve **Top-10** from the filtered pool.  
* Use **Cross-Encoder** to rerank and output **Top-5**.  
3. **Auto-map Gate**  
   Return one of these decisions:  
* `AUTO_MATCH`: only if ALL gates pass:  
  * **Margin Gate**: `(Score1 - Score2) >= Margin`  
    * `Margin` is configurable (**start with 0.1**, then tune on validation)  
  * **Team/Category Gate**: block category mismatches using rule-based tokens  
    * e.g., `[WOMEN] [U23] [RESERVES] [B-TEAM]`  
  * **Min Score Gate**: `Score1 >= MinScore` (configurable)  
* Otherwise return `NEED_REVIEW`.  
4. **Deliver results to Platform**  
* Write the inference result to Platform (DB or API) using the standard JSON below.  
* **Must include `mapping_suggestion_id`** (unique session ID).

---

### **Step 3 — Admin Verification UI (Platform Team)**

**Platform Team**

* Create two queues:  
  * **Review Queue**: items with `NEED_REVIEW`  
  * **Auto-Approve Queue**: items with `AUTO_MATCH`  
    * Do **NOT** sync AUTO\_MATCH directly to production at first. Use this queue for sampling verification.

**UI Workflow**

* Reviewer clicks one OP match → system shows Top-5 candidates (snapshot)  
* Reviewer selects one candidate → clicks:  
  * **\[Match\]** or **\[Swapped\]**  
* If none are correct → click:  
  * **\[No Match Found\]**

**Important UX Rule**

* Ensure flow is always: **\[Select Candidate\] → \[Decision\]**

---

### 

### **Step 4 — Feedback & Production Sync (Platform → AI)**

When the reviewer submits a decision, Platform must do TWO actions immediately:

1. **Production Sync (Platform Truth)**  
* Update production mapping ONLY for:  
  * `CONFIRMED` (Match/Swapped)  
  * or approved AUTO\_MATCH cases (if your process allows)  
* Enforce strict state tracking to avoid duplicates.  
2. **Send Feedback to AI**  
* Call AI feedback endpoint with:  
  * `mapping_suggestion_id`  
  * decision: `MATCH | SWAPPED | NO_MATCH |UNCERTAIN(Pending) | PARTIAL_MATCH`  
  * selected `bet365_match_id` (for MATCH/SWAPPED)  
  * if NO\_MATCH: include `reason_code`:  
    * `NO_MATCH_NOT_IN_B365` (match truly not present in Bet365 / feed delay)  
    * `NO_MATCH_NOT_IN_TOP5` (match exists but was not retrieved in Top-5)

---

### **Step 5 — Retraining & Optimization (AI Team)**

**AI Team**

* Store all feedback as training data:  
  * `MATCH/SWAPPED` → new positives  
  * `NO_MATCH` → store as **Unmatched Case**  
  * candidates shown in a NO\_MATCH session → harvest as **hard negatives**  
* **Hard Negative Mining rules**  
  * Negatives must come from same sport \+ ±30 minutes window  
  * Strictly exclude the true positive (`bet365_match_id`) to avoid label contamination  
* **SBERT Training**  
  * Train SBERT as a retrieval model using **MultipleNegativesRankingLoss**  
  * Use `[OP_text, B365_text]` pairs (never `[text, text]`)  
* Implement **Lock Logic**  
  * Do not reprocess matches already `CONFIRMED` or `UNMATCHED`.

---

## **3\) Key Must-Haves (Non-Negotiable)**

### **AI Team **

* SBERT training: **MultipleNegativesRankingLoss** with `[OP, B365]` pairs  
* Rule-based **Category Tokens** in input strings: `[WOMEN] [U23] [RESERVES] [B-TEAM]`  
* Candidate pre-filter: **Same sport \+ kickoff ±30 min**  
* Auto-map gates:  
  * tunable **Margin** (start 0.1), **Team/Category gate**, optional **MinScore**  
* **No accidental positives** in negative pool  
* Output must include **`mapping_suggestion_id`**

### **Platform Team**

* Admin UI:  
  * “Select one of Top-5 → Match/Swapped” OR “No Match Found”  
  * Separate **Auto-Approve Queue** (do not instantly push AUTO\_MATCH to production initially)  
* State tracking:  
  * `NEW → PREDICTED → (AUTO_MATCH / NEED_REVIEW) → (CONFIRMED / UNMATCHED)`  
* Enforce DB-level 1:1 rule if possible (recommended):  
  * one B365 match should not be confirmed for multiple OP matches

---

## **4\) Standard Integration JSON (AI → Platform)**

{  
  "mapping\_suggestion\_id": "uuid-or-int",  
  "oddsportal\_match": {  
    "match\_id": "...",  
    "sport": "Soccer",  
    "home": "...",  
    "away": "...",  
    "kickoff\_time": 1768429800  
  },  
  "candidates\_top5": \[  
    {  
      "bet365\_match\_id": "...",  
      "home": "...",  
      "away": "...",  
      "score": 0.98,  
      "time\_diff\_min": 5,  
      "is\_swapped\_possible": true  
    }  
  \],  
  "auto\_decision": "AUTO\_MATCH | NEED\_REVIEW",  
  "debug": {  
    "score1": 0.98,  
    "score2": 0.90,  
    "margin": 0.08,  
    "team\_gate": "PASSED"  
  }  
}

---

## **5\) Feedback Payload (Platform → AI)**

{  
  "mapping\_suggestion\_id": "uuid-or-int",  
  "decision": "MATCH | SWAPPED | NO\_MATCH",  
  "selected\_bet365\_match\_id": "...",  
  "reason\_code": "NO\_MATCH\_NOT\_IN\_B365 | NO\_MATCH\_NOT\_IN\_TOP5"  
}

---

원하면, 이 문서를 **AI팀용(파이썬 작업만)** / **플랫폼팀용(UI+백엔드만)** 으로 2개 메시지로 더 짧게 쪼개서도 만들어줄게.


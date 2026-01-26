# CPS Annotation Evaluation Report

**Model**: `gpt-4o-mini`
**Generated**: 2026-01-22 09:47:13
**Total Samples**: 2,410

## Overall Metrics Summary

| Facet | N | Kappa | Precision | Recall | F1 | Accuracy |
|-------|---|-------|-----------|--------|----|---------| 
| CSK | 2,410 | 0.0055 | 0.5294 | 0.0098 | 0.0193 | 0.6203 |
| NC | 2,410 | 0.0273 | 0.6944 | 0.0287 | 0.0552 | 0.6448 |
| MTF | 2,409 | 0.0090 | 0.1429 | 0.0570 | 0.0814 | 0.8311 |

---

## Multi-Label Confusion Matrix

This shows how all 8 possible label combinations `[CSK, NC, MTF]` get confused with each other.

![Multi-Label Confusion Matrix](confusion_matrix_multilabel.png)

### Classification Flow (Multi-Label)

Green flows indicate correct predictions, red flows indicate misclassifications.

![Multi-Label Sankey Diagram](sankey_multilabel.png)

---

## Detailed Per-Facet Analysis

### Constructing Shared Knowledge (CSK)

**Label Distribution:**
- Ground Truth: 916 positive (38.0%), 1,494 negative (62.0%)
- Predicted: 17 positive (0.7%), 2,393 negative (99.3%)

**Confusion Matrix Breakdown:**
| | Predicted Negative | Predicted Positive |
|---|---|---|
| **GT Negative** | TN: 1,486 | FP: 8 |
| **GT Positive** | FN: 907 | TP: 9 |

![Confusion Matrix - CSK](confusion_matrix_CSK.png)

![Sankey Diagram - CSK](sankey_CSK.png)


### Negotiation & Coordination (NC)

**Label Distribution:**
- Ground Truth: 870 positive (36.1%), 1,540 negative (63.9%)
- Predicted: 36 positive (1.5%), 2,374 negative (98.5%)

**Confusion Matrix Breakdown:**
| | Predicted Negative | Predicted Positive |
|---|---|---|
| **GT Negative** | TN: 1,529 | FP: 11 |
| **GT Positive** | FN: 845 | TP: 25 |

![Confusion Matrix - NC](confusion_matrix_NC.png)

![Sankey Diagram - NC](sankey_NC.png)


### Maintaining Team Function (MTF)

**Label Distribution:**
- Ground Truth: 317 positive (13.2%), 2,092 negative (86.8%)
- Predicted: 128 positive (5.3%), 2,281 negative (94.7%)

**Confusion Matrix Breakdown:**
| | Predicted Negative | Predicted Positive |
|---|---|---|
| **GT Negative** | TN: 1,984 | FP: 108 |
| **GT Positive** | FN: 298 | TP: 18 |

![Confusion Matrix - MTF](confusion_matrix_MTF.png)

![Sankey Diagram - MTF](sankey_MTF.png)


---

## Misclassification Analysis

### Constructing Shared Knowledge (CSK)

- **Total False Positives (FP)**: 8
- **Total False Negatives (FN)**: 907

#### False Positive Examples (GT=0, Pred=1)

**Sample 2:**
> The task that you are doing today is called the weights task

*Model Reasoning*: Participant 4 is describing the task, which helps build a shared understanding of the challenge situation.

**Sample 451:**
> So how about yellow

*Model Reasoning*: Participant 3 is suggesting a new idea about determining the weight of the yellow block, which aligns with the task of identifying weights, thus falling under CSK.

**Sample 577:**
> What if

*Model Reasoning*: Participant 1's utterance 'What if' suggests a potential idea or scenario for consideration, indicating an attempt to construct shared knowledge about the challenge situation.

**Sample 1032:**
> If we decide to continue on the pattern

*Model Reasoning*: Participant 1 is suggesting to follow a specific pattern in their estimation process, which relates to discussing the challenge situation and how they plan to approach the task.

**Sample 1164:**
> Let's see how callibrated this is or

*Model Reasoning*: Participant 1 is discussing the challenge situation by questioning the calibration of the scale, which relates to the game mechanics.

#### False Negative Examples (GT=1, Pred=0)

**Sample 12:**
> Oh and we don't touch the scale yet

*Model Reasoning*: Failed to parse

**Sample 16:**
> Ok so you want us to

*Model Reasoning*: The utterance is neutral and does not provide specific information related to the task or the group dynamics.

**Sample 17:**
> Use the scale to determine the weights ok cool

*Model Reasoning*: Failed to parse

**Sample 19:**
> Where's the survey oh ok

*Model Reasoning*: Failed to parse

**Sample 27:**
> Just like put it on

*Model Reasoning*: Participant 1's utterance seems to be a suggestion related to the task at hand, indicating an action to take ('put it on'). This can be categorized under Maintaining Team Function (MTF) as it is likely a request for collaboration or instruction.


### Negotiation & Coordination (NC)

- **Total False Positives (FP)**: 11
- **Total False Negatives (FN)**: 845

#### False Positive Examples (GT=0, Pred=1)

**Sample 99:**
> You're right

*Model Reasoning*: Participant 1 is acknowledging Participant 2's previous statement about the weights of the blocks, which demonstrates agreement and support for the ongoing discussion.

**Sample 583:**
> Yes  

*Model Reasoning*: Participant 1's response 'Yes' acknowledges the previous statements made by Participant 3 and aligns with their ideas, which fits the criteria of responding to others' ideas.

**Sample 784:**
> True  

*Model Reasoning*: Participant 1's response 'True' acknowledges the statement made by Participant 2 about weight being related to physics, which indicates agreement and supports the ongoing discussion about measuring force.

**Sample 1259:**
> Did it

*Model Reasoning*: Participant 2's utterance 'Did it' appears to confirm that they have completed a task or action, which aligns with discussing the results of their collaboration. It does not construct shared knowledge or maintain team function.

**Sample 1280:**
> Yeah

*Model Reasoning*: Participant 3's utterance 'Yeah' acknowledges and agrees with the previous statements made by Participants 1 and 2, indicating a response to others' ideas.

#### False Negative Examples (GT=1, Pred=0)

**Sample 29:**
> Hold on lets just uh

*Model Reasoning*: Failed to parse

**Sample 30:**
> It seems pretty balanced

*Model Reasoning*: Failed to parse

**Sample 31:**
> Yeah ok so now we know that this is also ten

*Model Reasoning*: Failed to parse

**Sample 36:**
> And then see if there's thats twenty which is not

*Model Reasoning*: Failed to parse

**Sample 37:**
> It's more

*Model Reasoning*: Failed to parse


### Maintaining Team Function (MTF)

- **Total False Positives (FP)**: 108
- **Total False Negatives (FN)**: 298

#### False Positive Examples (GT=0, Pred=1)

**Sample 1:**
> Alright so

*Model Reasoning*: The utterance indicates a transition to a new topic or point in the discussion, suggesting that Participant 4 is maintaining team function by preparing to engage further in the conversation.

**Sample 20:**
> Um it's turned off

*Model Reasoning*: Participant 3's utterance provides information about the status of the survey, which is relevant to the team's task and indicates a need for coordination regarding the survey process.

**Sample 23:**
> Ok

*Model Reasoning*: Participant 2's utterance 'Ok' is a minimal acknowledgment, which falls under maintaining team function as it indicates engagement with the conversation.

**Sample 27:**
> Just like put it on

*Model Reasoning*: Participant 1's utterance seems to be a suggestion related to the task at hand, indicating an action to take ('put it on'). This can be categorized under Maintaining Team Function (MTF) as it is likely a request for collaboration or instruction.

**Sample 35:**
> Cool

*Model Reasoning*: Participant 2's response 'Cool' is a supportive acknowledgment of the conversation, indicating agreement or encouragement, which fits under maintaining team function (MTF-2). It does not provide any new content or logical reasoning related to the task.

#### False Negative Examples (GT=1, Pred=0)

**Sample 83:**
> And green is twenty ok so which one is blue and which one is purple

*Model Reasoning*: Failed to parse

**Sample 93:**
> Yes do you want to check real quick

*Model Reasoning*: Failed to parse

**Sample 119:**
> Yeah try picking these two up and you can close your eyes and try to hold it from the

*Model Reasoning*: Failed to parse

**Sample 121:**
> Yeah same point maybe

*Model Reasoning*: Participant 1's utterance 'Yeah same point maybe' does not contribute to constructing shared knowledge, negotiating or coordinating ideas, or maintaining team function. It appears to be a neutral response without a clear function in the discussion.

**Sample 125:**
> It still feels heavier but it could just be

*Model Reasoning*: Failed to parse


---

## Summary Statistics

- **Exact Match Rate** (all 3 facets match): 26.02% (627/2,410)

**Facet Performance Ranking (by Kappa):**
1. NC: κ = 0.0273
2. MTF: κ = 0.0090
3. CSK: κ = 0.0055

**Parse Failures**: 1,950 (80.9%)

# Short Answers

### If you only had 200 labeled replies, how would you improve the model without collecting thousands more?
- Use data augmentation (back-translation, synonym replacement), leverage pre-trained transformers and few-shot learning, and use cross-validation and careful regularization. Also consider active learning to label the most informative examples.

### How would you ensure your reply classifier doesn’t produce biased or unsafe outputs in production?
- Monitor predictions for demographic or content-based bias, add guardrails that detect and flag offensive content, perform human-in-the-loop review for low-confidence predictions, and log model inputs/outputs for auditing.

### Suppose you want to generate personalized cold email openers using an LLM. What prompt design strategies would you use to keep outputs relevant and non-generic?
- Provide context (recipient role, company, short background), ask for 3 variants with different tones, include constraints (length, avoid cliches), and use examples (few-shot) showing good and bad openers.

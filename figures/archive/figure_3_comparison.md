# Figure 3: Response Comparison Examples

## Prompt:
Calculate step by step: If a store offers a 15% discount on a $280 item, and then applies an 8.5% sales tax on the discounted price, what is the final amount the customer pays?

---

### Base Model Response:
```
Also, check if the tax is applied after the discount or before?

Let me try again. First, find the discounted price of the item. The original price is $280, and the discount is 15%, so the discount amount is 280 * 15%. Then subtract that from the original price to get the discounted price. After that, apply the 8.5% sales tax on the discounted price. What is the final amount the customer pays? [...]
```
*Length: 2846 chars, 554 words*

### Fine-tuned (5-epoch) Response:
```
Additionally, how many times has this pattern occurred in the past 30 years?
Answer:
The customer pays $280 - 15% discount = $219. Then, applying an 8.5% sales tax on the discounted price:
219 * 8.5% = $18.655
Total amount paid: $219 + $18.655 = $237.655

The pattern occurred 30 times in the past 30 years.

**Final Answer:**
- The customer pays approximately \$237.66. [...]
```
*Length: 2873 chars, 389 words*

### Key Differences:
- **Length reduction**: 2846 → 2873 chars (-0.9% reduction)
- **Reasoning words**: 20 → 3 (85.0% reduction)

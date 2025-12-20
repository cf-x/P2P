# Instruction

Generate a targeted product rating user profile for **product review rating prediction** based on the provided product purchase and review history data. This profile will be used to predict what rating (1-5 stars) this user would assign to product reviews based on sentiment and satisfaction levels.

**IMPORTANT: You must analyze HOW this user rates and evaluates products, NOT list WHAT products they buy. Do not output lists of product names, brands, or categories. Focus only on rating behavior patterns and evaluation tendencies.**

Focus on understanding rating decision patterns:

1. **Rating Scale Usage and Scoring Patterns:**
   - Tendency toward harsh vs. generous rating assignments (1-5 scale usage patterns)
   - Consistency in rating similar products and experiences
   - Sensitivity to different aspects affecting rating decisions (quality, value, experience)
   - Correlation between review sentiment and numerical rating assignments

2. **Product Evaluation Criteria and Weighting:**
   - Emphasis on product quality vs. price value in rating decisions
   - Importance of shipping, customer service, and overall experience in ratings
   - Tolerance for product defects and issues affecting rating severity
   - Brand loyalty influence on rating generosity or critical assessment

3. **Review Analysis and Rating Behavioral Patterns:**
   - Ability to interpret review sentiment and translate to numerical ratings
   - Handling of mixed reviews with both positive and negative elements
   - Response to detailed vs. brief reviews in rating assessment
   - Consistency in rating across different product categories

4. **Consumer Psychology and Rating Calibration:**
   - Personal satisfaction thresholds affecting rating boundaries
   - Expectation management and how it influences rating decisions
   - Social influence and comparison to other reviews in rating choices
   - Evolution of rating standards over time and experience


# User History Data

{{ user_history }}

# Output Format

Output the consumer user profile strictly in plain text describing the user's rating patterns and product evaluation behavior tendencies. Focus specifically on patterns that predict **what numerical rating (1-5) this user would assign to product reviews** based on their consumption patterns and evaluation criteria.

**Do NOT output:**
- Lists of product names or brands
- Product categories or types
- Specific purchases or items
- Shopping content examples

**DO output:**
- Rating behavior patterns and preferences
- Product evaluation criteria and tendencies
- Consumer decision-making behavioral patterns
- Satisfaction threshold and scoring patterns

Derive insights strictly from the provided product interaction and review data. Do not include explanations, introductions, headings, bullet points, or any formatting structure. 
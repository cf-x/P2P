# Instruction

Generate a targeted product review writing user profile for **detailed product review generation** based on the provided product review history and rating patterns. This profile will be used to predict what style of product reviews this user would write given ratings, product descriptions, and review summaries.

**IMPORTANT: You must analyze HOW this user writes product reviews, NOT list WHAT products they review. Do not output lists of product names, brands, or categories. Focus only on review writing style patterns and behavioral tendencies.**

Focus on understanding review writing adaptation patterns:

1. **Review Writing Style and Personal Voice:**
   - Characteristic writing tone and personality in review composition
   - Use of descriptive vs. analytical language in product evaluation
   - Balance between personal experience and objective product assessment
   - Consistent voice elements that persist across different product types

2. **Product Review Content Development and Structure:**
   - Approach to expanding summary information into detailed review text
   - Integration of product features with personal usage experiences
   - Use of comparative analysis and contextual product positioning
   - Organization and flow preferences in review structure

3. **Review Generation Behavioral Patterns:**
   - Adaptation of review depth based on rating levels (detailed for extreme ratings)
   - Consistency between numerical ratings and written review sentiment
   - Handling of product disappointments vs. positive experiences in review tone
   - Integration of technical details vs. practical usage focus

4. **Consumer Perspective and Review Authenticity Indicators:**
   - Personal experience authentication and credibility markers
   - Use of specific examples and practical usage scenarios
   - Emotional expression patterns related to product satisfaction levels
   - Helpfulness orientation and consideration for other potential buyers


# User History Data

{{ user_history }}

# Output Format

Output the product review user profile strictly in plain text describing the user's review writing patterns and product evaluation behavior tendencies. Focus specifically on patterns that predict **what style of detailed product reviews this user would write** based on rating information and product characteristics.

**Do NOT output:**
- Lists of product names or brands
- Product categories or types
- Specific reviews or purchase examples
- Shopping content keywords

**DO output:**
- Review writing style patterns and preferences
- Product evaluation behavioral patterns
- Consumer perspective and authenticity markers
- Review composition and structure tendencies

Derive insights strictly from the provided product review and interaction data. Do not include explanations, introductions, headings, bullet points, or any formatting structure. 
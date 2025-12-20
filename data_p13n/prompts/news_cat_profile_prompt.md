# Instruction

Generate a targeted news classification user profile for **news article categorization into topical categories** based on the provided news reading and interaction history data. This profile will be used to predict which categories (travel, education, politics, sports, business, entertainment, etc.) a user would assign to news articles.

**IMPORTANT: You must analyze HOW this user categorizes and classifies news articles, NOT list WHAT news topics they read. Do not output lists of news topics, categories, or keywords. Focus only on categorization behavior patterns and decision-making tendencies.**

Focus on understanding categorization decision patterns:

1. **Category Recognition and Assignment Patterns:**
   - Expertise in distinguishing between specific news categories: travel, education, parents, style & beauty, entertainment, food & drink, science & technology, business, sports, healthy living, women, politics, crime, culture & arts, religion
   - Consistency in applying category labels to similar content types
   - Ability to handle ambiguous articles that could fit multiple categories
   - Preference for primary vs. secondary category assignment in multi-topic articles

2. **Content Analysis for Classification Decisions:**
   - Focus on headlines vs. full article content for categorization
   - Recognition of subtle topic indicators and keyword patterns
   - Handling of opinion pieces vs. news reporting in categorization
   - Sensitivity to context and framing that affects category assignment

3. **News Classification Behavioral Patterns:**
   - Systematic vs. intuitive approach to category assignment
   - Bias toward certain categories based on personal interests or expertise
   - Adaptation to evolving news categories and emerging topics
   - Consistency across different news sources and writing styles

4. **Topical Domain Knowledge and Classification Accuracy:**
   - Areas of high categorization confidence and expertise
   - Recognition of specialized subcategories within broader topics
   - Understanding of category boundaries and overlapping themes
   - Influence of personal reading habits on classification decisions


# User History Data

{{ user_history }}

# Output Format

Output the news categorization user profile strictly in plain text describing the user's news classification patterns and categorization behavior tendencies. Focus specifically on patterns that predict **which topical category this user would assign to news articles** based on their news consumption and classification tendencies.

**Do NOT output:**
- Lists of news topics or categories
- Current events or news keywords
- Political topics or specific news content
- Names of news sources or publications

**DO output:**
- Categorization behavior patterns and preferences
- News classification criteria and tendencies
- Content analysis behavioral patterns
- Category assignment decision-making patterns

Derive insights strictly from the provided news reading and interaction data. Do not include explanations, introductions, headings, bullet points, or any formatting structure. 
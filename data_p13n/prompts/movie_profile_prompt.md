# Instruction

Generate a targeted movie preference user profile for **movie tag prediction and genre classification** based on the provided movie viewing and rating history data. This profile will be used to predict which tags (sci-fi, comedy, action, romance, etc.) a user would associate with specific movies.

**IMPORTANT: You must analyze HOW this user categorizes and tags movies, NOT list WHAT movies they watch. Do not output lists of movie titles, genres, or actors. Focus only on categorization behavior patterns and tagging tendencies.**

Focus on understanding tag association patterns:

1. **Genre and Tag Recognition Patterns:**
   - Sensitivity to specific movie genres and thematic elements (sci-fi, comedy, action, romance, thriller)
   - Ability to identify complex themes (dystopia, psychology, social commentary, twist endings)
   - Preference for broad genre categories vs. nuanced thematic tags
   - Consistency in tag association across similar movie types

2. **Movie Categorization and Classification Preferences:**
   - Emphasis on plot elements vs. stylistic elements in tag selection
   - Recognition of hybrid genres and multi-tag movies
   - Preference for mainstream tags vs. artistic/niche category labels
   - Influence of personal taste on objective movie categorization

3. **Cinematic Pattern Recognition Behavioral Patterns:**
   - Focus on narrative structure vs. production values in tag assignment
   - Sensitivity to cultural/temporal movie characteristics
   - Adaptation to evolving genre definitions and new movie categories
   - Bias toward familiar vs. experimental movie categorization

4. **Movie Evaluation and Tagging Indicators:**
   - Critical analysis depth affecting tag precision
   - Personal movie experience informing tag selection accuracy
   - Genre expertise areas with higher tagging confidence
   - Demographic influences on movie tag perception and assignment

# User History Data

{{ user_history }}

# Output Format

Output the entertainment user profile strictly in plain text describing the user's movie categorization patterns and tagging behavior tendencies. Focus specifically on patterns that predict **which tags this user would assign to movies** based on their viewing preferences and categorization tendencies.

**Do NOT output:**
- Lists of movie titles or names
- Genre lists or actor names
- Specific movies they've watched
- Entertainment content examples

**DO output:**
- Categorization behavior patterns and preferences
- Tag assignment criteria and tendencies
- Movie evaluation behavioral patterns
- Genre recognition and classification patterns

Derive insights strictly from the provided movie viewing and rating interaction data. Do not include explanations, introductions, headings, bullet points, or any formatting structure. 
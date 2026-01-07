pip freeze > requirements.txt
tree -I 'venv|data|sorted_photos|review_collages|archive|__pycache__|.git' --dirsfirst> structure.txt
lazy_git
##
##
##graph TD
##    A[📧 Email Harvest] -->|Scan Inbox| B(URL List)
##    B -->|Fetch| C{Scraper Gen 205}
##    C -->|Attempt 1| D[Standard Request]
##    D -->|Fail?| E[Robust Session + Headers]
##    E -->|Success| F[Sanitize HTML <br/> Remove Null Bytes]
##    F -->|Parse Fail?| G[Naive Fallback <br/> Brute Force p-tags]
##    G --> H[Clean Text]
##    F -->|Success| H
##    H --> I[🤖 AI Clustering]
##    I -->|1 Item?| J[Single Article Mode]
##    I -->|Many Items?| K[Cluster & Summarize]
##    J --> L[📝 Script Generation]
##    K --> L
##    L --> M[🎧 Audio Production <br/> EdgeTTS + Music]
##    M --> N[🚀 GitHub Deployment <br/> RSS Update]
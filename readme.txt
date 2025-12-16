pip freeze > requirements.txt
tree -I 'venv|data|sorted_photos|review_collages|archive|__pycache__|.git' --dirsfirst> structure.txt
lazy_git
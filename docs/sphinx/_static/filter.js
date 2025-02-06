document.addEventListener('DOMContentLoaded', function() {
    const notebooks = document.querySelectorAll('.notebook-entry');
    const tagButtons = document.querySelectorAll('.tag-button');

    tagButtons.forEach(button => {
        button.addEventListener('click', function() {
            const tag = this.getAttribute('data-tag');
            
            tagButtons.forEach(btn => btn.classList.remove('active'));
            this.classList.add('active');
            
            notebooks.forEach(notebook => {
                const tags = notebook.getAttribute('data-tags').split(',');
                notebook.style.display = tag === 'all' || tags.includes(tag) ? 'block' : 'none';
            });
        });
    });
});


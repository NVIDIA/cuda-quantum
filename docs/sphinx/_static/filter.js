document.addEventListener('DOMContentLoaded', function() {
    const notebooks = document.querySelectorAll('.notebook-entry');
    const tagButtons = document.querySelectorAll('.tag-button');
    
    // Track active filters for each group
    const activeFilters = {
        domain: 'all',
        backend: 'all',
        library: 'all',
        occasion: 'all'
    };

    tagButtons.forEach(button => {
        button.addEventListener('click', function() {
            const group = this.getAttribute('data-group');
            const tag = this.getAttribute('data-tag');
            
            // Update active button within the same group
            document.querySelectorAll(`.tag-button[data-group="${group}"]`)
                .forEach(btn => btn.classList.remove('active'));
            this.classList.add('active');
            
            // Update active filters
            activeFilters[group] = tag;
            
            // Apply all filters
            notebooks.forEach(notebook => {
                const notebookTags = notebook.getAttribute('data-tags').split(',').map(t => t.trim());
                const isVisible = Object.entries(activeFilters).every(([group, tag]) => {
                    return tag === 'all' || notebookTags.includes(tag);
                });
                notebook.style.display = isVisible ? 'block' : 'none';
            });
        });
    });
});


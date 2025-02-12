document.addEventListener('DOMContentLoaded', function() {
    const notebooks = document.querySelectorAll('.notebook-entry');
    const tagButtons = document.querySelectorAll('.tag-button');

    const activeFilters = {
        domain: 'all',
        backend: 'all',
        occasion: 'all',
        subFilter: 'all'
    };

    const backendCategories = {
        noiseless: ['gpu', 'mgpu', 'mqpu'],
        noisy: ['density'],
        qpu: ['neutral']
    };

    tagButtons.forEach(button => {
        button.addEventListener('click', function() {
            const group = this.getAttribute('data-group');
            const tag = this.getAttribute('data-tag');

            if (this.classList.contains('sub-option')) {
                if (activeFilters.subFilter === tag) {
                    activeFilters.subFilter = 'all';
                    activeFilters.backend = 'all';
                    document.querySelectorAll('.sub-option, .backend-toggle').forEach(btn => 
                        btn.classList.remove('active'));
                    document.querySelector('.tag-button[data-tag="all"]').classList.add('active');
                } else {
                    activeFilters.subFilter = tag;
                    activeFilters.backend = this.closest('.backend-group')
                        .querySelector('.backend-toggle')
                        .getAttribute('data-tag');

                    document.querySelectorAll('.sub-option').forEach(btn =>
                        btn.classList.toggle('active', btn.getAttribute('data-tag') === tag));

                    document.querySelectorAll('.backend-toggle').forEach(toggle =>
                        toggle.classList.toggle('active',
                            toggle.getAttribute('data-tag') === activeFilters.backend));
                    
                    document.querySelector('.tag-button[data-tag="all"]').classList.remove('active');
                }
            } else if (this.classList.contains('backend-toggle')) {
                if (activeFilters.backend === tag) {
                    activeFilters.backend = 'all';
                    activeFilters.subFilter = 'all';
                    document.querySelectorAll('.backend-toggle, .sub-option').forEach(btn =>
                        btn.classList.remove('active'));
                    document.querySelector('.tag-button[data-tag="all"]').classList.add('active');
                } else {
                    activeFilters.backend = tag;
                    activeFilters.subFilter = 'all';
                    document.querySelectorAll('.backend-toggle').forEach(btn =>
                        btn.classList.toggle('active', btn.getAttribute('data-tag') === tag));
                    document.querySelectorAll('.sub-option').forEach(btn =>
                        btn.classList.remove('active'));
                    document.querySelector('.tag-button[data-tag="all"]').classList.remove('active');
                }
            } else {
                // Handle regular filter buttons
                if (group === 'occasion') {
                    if (activeFilters[group] === tag) {
                        // If clicking the same tag, reset to 'all'
                        activeFilters[group] = 'all';
                        this.classList.remove('active');
                    } else {
                        activeFilters[group] = tag;
                        document.querySelectorAll(`.tag-button[data-group="${group}"]`)
                            .forEach(btn => btn.classList.remove('active'));
                        this.classList.add('active');
                    }
                } else {
                    document.querySelectorAll(`.tag-button[data-group="${group}"]`)
                        .forEach(btn => btn.classList.remove('active'));
                    this.classList.add('active');
                    activeFilters[group] = tag;
                }

                if (group === 'backend') {
                    activeFilters.subFilter = 'all';
                }
            }

            // Apply all filters
            notebooks.forEach(notebook => {
                const notebookTags = notebook.getAttribute('data-tags').split(',').map(t => t.trim());
                const isVisible = Object.entries(activeFilters).every(([group, tag]) => {
                    if (group === 'subFilter' && tag !== 'all') {
                        return notebookTags.includes(tag);
                    }
                    return tag === 'all' || notebookTags.includes(tag);
                });
                notebook.style.display = isVisible ? 'grid' : 'none';
            });
        });
    });
});

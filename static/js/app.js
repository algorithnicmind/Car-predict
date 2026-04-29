/**
 * CarValue AI — Frontend Application Logic
 *
 * Handles metadata loading, form population, validation,
 * and prediction API calls.
 */

(function () {
    'use strict';

    // ── DOM References ──
    const form           = document.getElementById('predict-form');
    const btnPredict     = document.getElementById('btn-predict');
    const resultContainer = document.getElementById('result-container');
    const resultPrice    = document.getElementById('result-price');
    const errorContainer = document.getElementById('error-container');
    const errorMessage   = document.getElementById('error-message');
    const heroR2         = document.getElementById('hero-r2');
    const statR2         = document.getElementById('stat-r2');
    const statMAE        = document.getElementById('stat-mae');
    const statRMSE       = document.getElementById('stat-rmse');
    const barR2          = document.getElementById('bar-r2');

    // ── State ──
    let metadata = null;

    // ── Utilities ──
    function formatCurrency(value) {
        const num = Math.round(value);
        return 'Rs. ' + num.toLocaleString('en-IN');
    }

    function showError(msg) {
        errorMessage.textContent = msg;
        errorContainer.classList.add('show');
        resultContainer.classList.remove('show');
        setTimeout(() => errorContainer.classList.remove('show'), 6000);
    }

    function hideMessages() {
        errorContainer.classList.remove('show');
    }

    function animatePrice(targetValue) {
        const duration = 1200;
        const startTime = performance.now();
        const startValue = 0;

        function update(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);

            // Ease-out cubic
            const eased = 1 - Math.pow(1 - progress, 3);
            const currentValue = startValue + (targetValue - startValue) * eased;

            resultPrice.textContent = formatCurrency(currentValue);

            if (progress < 1) {
                requestAnimationFrame(update);
            }
        }

        requestAnimationFrame(update);
    }

    // ── Load Metadata ──
    async function loadMetadata() {
        try {
            const res = await fetch('/api/metadata');
            if (!res.ok) throw new Error('Failed to load metadata');
            metadata = await res.json();
            populateDropdowns();
            populateHints();
            populateStats();
        } catch (err) {
            console.error('Metadata error:', err);
            showError('Could not load model data. Please refresh the page.');
        }
    }

    function populateDropdowns() {
        const catFeatures = metadata.categorical_features;

        for (const [col, values] of Object.entries(catFeatures)) {
            const select = document.getElementById(col);
            if (!select) continue;
            values.forEach(val => {
                const option = document.createElement('option');
                option.value = val;
                option.textContent = val;
                select.appendChild(option);
            });
        }
    }

    function populateHints() {
        const stats = metadata.feature_stats;
        for (const [feat, info] of Object.entries(stats)) {
            const hintEl = document.getElementById('hint-' + feat);
            if (!hintEl) continue;
            hintEl.textContent = `Range: ${info.min.toLocaleString()} – ${info.max.toLocaleString()}  |  Avg: ${info.mean.toLocaleString()}`;
        }
    }

    function populateStats() {
        const m = metadata.model_metrics;

        // Hero card
        heroR2.textContent = m.r2_percentage + '%';

        // Stats section
        statR2.textContent   = m.r2_percentage + '%';
        statMAE.textContent  = 'Rs. ' + Math.round(m.mae).toLocaleString('en-IN');
        statRMSE.textContent = 'Rs. ' + Math.round(m.rmse).toLocaleString('en-IN');

        // Animate bar
        setTimeout(() => {
            barR2.style.width = m.r2_percentage + '%';
        }, 500);
    }

    // ── Prediction ──
    async function predict(payload) {
        btnPredict.classList.add('loading');
        btnPredict.disabled = true;
        hideMessages();

        try {
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });

            const data = await res.json();

            if (!res.ok) {
                throw new Error(data.error || 'Prediction failed');
            }

            // Show result with animated counter
            resultContainer.classList.add('show');
            animatePrice(data.predicted_price);

            // Scroll to result
            resultContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });

        } catch (err) {
            showError(err.message);
        } finally {
            btnPredict.classList.remove('loading');
            btnPredict.disabled = false;
        }
    }

    // ── Form Submit ──
    form.addEventListener('submit', function (e) {
        e.preventDefault();

        const numFeatures = metadata.numerical_features;
        const catFeatures = Object.keys(metadata.categorical_features);
        const payload = {};

        // Collect numerical
        for (const feat of numFeatures) {
            const input = document.getElementById(feat);
            if (!input || input.value === '') {
                showError(`Please enter a value for ${feat.replace(/_/g, ' ')}`);
                input?.focus();
                return;
            }
            payload[feat] = parseFloat(input.value);
        }

        // Collect categorical
        for (const feat of catFeatures) {
            const select = document.getElementById(feat);
            if (!select || select.value === '') {
                showError(`Please select a ${feat.replace(/_/g, ' ')}`);
                select?.focus();
                return;
            }
            payload[feat] = select.value;
        }

        predict(payload);
    });

    // ── Smooth scroll for nav links ──
    document.querySelectorAll('.nav-link, .hero-cta').forEach(link => {
        link.addEventListener('click', function (e) {
            const href = this.getAttribute('href');
            if (href && href.startsWith('#')) {
                e.preventDefault();
                const target = document.querySelector(href);
                if (target) {
                    target.scrollIntoView({ behavior: 'smooth' });
                }

                // Update active state
                document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
                if (this.classList.contains('nav-link')) {
                    this.classList.add('active');
                }
            }
        });
    });

    // ── Navbar scroll effect ──
    let lastScroll = 0;
    window.addEventListener('scroll', () => {
        const navbar = document.getElementById('navbar');
        const scrollY = window.scrollY;
        if (scrollY > 100) {
            navbar.style.background = 'rgba(10, 10, 15, 0.92)';
        } else {
            navbar.style.background = 'rgba(10, 10, 15, 0.7)';
        }
        lastScroll = scrollY;
    });

    // ── Intersection Observer for scroll animations ──
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px',
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    document.querySelectorAll('.step-card, .stat-card').forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(30px)';
        el.style.transition = 'opacity 0.6s ease-out, transform 0.6s ease-out';
        observer.observe(el);
    });

    // ── Input focus animations ──
    document.querySelectorAll('.form-input').forEach(input => {
        input.addEventListener('focus', function() {
            this.closest('.form-group').style.transform = 'translateY(-2px)';
            this.closest('.form-group').style.transition = 'transform 0.2s ease';
        });
        input.addEventListener('blur', function() {
            this.closest('.form-group').style.transform = 'translateY(0)';
        });
    });

    // ── Reset button handler ──
    const btnReset = document.getElementById('btn-reset');
    if (btnReset) {
        btnReset.addEventListener('click', function () {
            resultContainer.classList.remove('show');
            errorContainer.classList.remove('show');
        });
    }

    // ── Scroll to Top ──
    const scrollTopBtn = document.getElementById('scroll-to-top');
    if (scrollTopBtn) {
        window.addEventListener('scroll', () => {
            if (window.scrollY > 500) {
                scrollTopBtn.classList.add('show');
            } else {
                scrollTopBtn.classList.remove('show');
            }
        });

        scrollTopBtn.addEventListener('click', () => {
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
    }

    // ── Init ──
    loadMetadata();

})();

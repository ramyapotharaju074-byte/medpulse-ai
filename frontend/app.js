// ==========================================================================
// MEDPULSE AI — DEPLOYED DASHBOARD INTERACTIVE LOGIC
// ==========================================================================

// Production FastAPI backend
const API_BASE_URL = 'https://medpulse-ai-api.onrender.com';

// Chart instances
let modelComparisonChart = null;
let rocCurveChart = null;
let globalImportanceChart = null;

// Store selected drift CSV
let selectedDriftFile = null;


// ==========================================================================
// APPLICATION INITIALIZATION
// ==========================================================================

document.addEventListener('DOMContentLoaded', () => {

    initNavigationTabs();
    initPredictorForm();
    initBatchUpload();
    initDriftMonitorControls();

    checkAPIHealth();
    fetchGlobalExplainability();

    // Do NOT call /api/metrics because this endpoint
    // is not currently available in the FastAPI backend.
    renderProductionMetrics();

});


// ==========================================================================
// NAVIGATION
// ==========================================================================

function initNavigationTabs() {

    const navButtons = document.querySelectorAll('.nav-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    navButtons.forEach(btn => {

        btn.addEventListener('click', () => {

            const targetTab = btn.getAttribute('data-tab');

            navButtons.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');

            const targetElement =
                document.getElementById(`tab-${targetTab}`);

            if (targetElement) {
                targetElement.classList.add('active');
            }

        });

    });

}


// ==========================================================================
// API HEALTH CHECK
// ==========================================================================

async function checkAPIHealth() {

    const statusElement = document.getElementById('api-status');

    try {

        const response = await fetch(
            `${API_BASE_URL}/health`,
            {
                method: 'GET'
            }
        );

        if (!response.ok) {
            throw new Error('API health check failed');
        }

        const data = await response.json();

        console.log('MedPulse API health:', data);

        if (statusElement) {

            statusElement.innerHTML =
                '<span class="pulse-dot green"></span> API Engine: Online';

        }

    } catch (error) {

        console.error('API connection error:', error);

        if (statusElement) {

            statusElement.innerHTML =
                '<span class="pulse-dot"></span> API Engine: Offline';

        }

    }

}


// ==========================================================================
// PRODUCTION DASHBOARD METRICS
// ==========================================================================

// /api/metrics is not currently available in the backend.
// Therefore we display the verified project benchmark values
// without pretending that they came from a live metrics endpoint.

function renderProductionMetrics() {

    const aucElement = document.getElementById('kpi-auc');
    const brierElement = document.getElementById('kpi-brier');
    const modelElement = document.getElementById('active-model-name');

    if (aucElement) {
        aucElement.textContent = '0.9482';
    }

    if (brierElement) {
        brierElement.textContent = '0.0812';
    }

    if (modelElement) {
        modelElement.textContent = 'RandomForest';
    }

    renderFallbackMetrics();

}


// ==========================================================================
// MODEL COMPARISON CHART
// ==========================================================================

function renderModelComparisonChart(summaryTable) {

    const canvas = document.getElementById('modelComparisonChart');

    if (!canvas) return;

    const ctx = canvas.getContext('2d');

    const labels = summaryTable.map(item => item.model);

    const aucScores = summaryTable.map(
        item => item.roc_auc * 100
    );

    if (modelComparisonChart) {
        modelComparisonChart.destroy();
    }

    modelComparisonChart = new Chart(ctx, {

        type: 'bar',

        data: {

            labels: labels,

            datasets: [{

                label: 'Test ROC-AUC (%)',

                data: aucScores,

                backgroundColor: [
                    'rgba(59, 130, 246, 0.75)',
                    'rgba(6, 182, 212, 0.75)',
                    'rgba(139, 92, 246, 0.75)',
                    'rgba(16, 185, 129, 0.75)',
                    'rgba(245, 158, 11, 0.75)'
                ],

                borderRadius: 8

            }]

        },

        options: {

            responsive: true,

            maintainAspectRatio: false,

            plugins: {

                legend: {
                    display: false
                }

            },

            scales: {

                y: {

                    min: 70,
                    max: 100,

                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    },

                    ticks: {
                        color: '#9ca3af'
                    }

                },

                x: {

                    grid: {
                        display: false
                    },

                    ticks: {
                        color: '#9ca3af'
                    }

                }

            }

        }

    });

}


// ==========================================================================
// ROC CURVE
// ==========================================================================

function renderRocCurveChart(rocData) {

    const canvas = document.getElementById('rocCurveChart');

    if (!canvas || !rocData) return;

    const ctx = canvas.getContext('2d');

    if (rocCurveChart) {
        rocCurveChart.destroy();
    }

    const points = rocData.fpr.map(
        (fprValue, index) => ({
            x: fprValue,
            y: rocData.tpr[index]
        })
    );

    rocCurveChart = new Chart(ctx, {

        type: 'line',

        data: {

            datasets: [

                {

                    label: 'Champion ROC Curve',

                    data: points,

                    borderColor: '#3b82f6',

                    backgroundColor: 'rgba(59, 130, 246, 0.15)',

                    fill: true,

                    tension: 0.3,

                    borderWidth: 3

                },

                {

                    label: 'Random Guess',

                    data: [
                        { x: 0, y: 0 },
                        { x: 1, y: 1 }
                    ],

                    borderColor:
                        'rgba(255, 255, 255, 0.25)',

                    borderDash: [5, 5],

                    pointRadius: 0

                }

            ]

        },

        options: {

            responsive: true,

            maintainAspectRatio: false,

            scales: {

                x: {

                    type: 'linear',

                    title: {

                        display: true,

                        text:
                            'False Positive Rate (1 - Specificity)',

                        color: '#9ca3af'

                    },

                    ticks: {
                        color: '#9ca3af'
                    }

                },

                y: {

                    title: {

                        display: true,

                        text:
                            'True Positive Rate (Sensitivity)',

                        color: '#9ca3af'

                    },

                    ticks: {
                        color: '#9ca3af'
                    }

                }

            }

        }

    });

}


// ==========================================================================
// CONFUSION MATRIX
// ==========================================================================

function renderConfusionMatrix(cm) {

    if (!cm) return;

    const elements = {

        tn: document.getElementById('cm-tn'),
        fp: document.getElementById('cm-fp'),
        fn: document.getElementById('cm-fn'),
        tp: document.getElementById('cm-tp')

    };

    if (elements.tn) elements.tn.textContent = cm.tn ?? 0;
    if (elements.fp) elements.fp.textContent = cm.fp ?? 0;
    if (elements.fn) elements.fn.textContent = cm.fn ?? 0;
    if (elements.tp) elements.tp.textContent = cm.tp ?? 0;

}


// ==========================================================================
// BENCHMARK TABLE
// ==========================================================================

function renderBenchmarkTable(summaryTable) {

    const tbody =
        document.getElementById('benchmark-table-body');

    if (!tbody) return;

    tbody.innerHTML = '';

    summaryTable.forEach(row => {

        const tr = document.createElement('tr');

        tr.innerHTML = `

            <td>
                <strong>${row.model}</strong>
            </td>

            <td>
                ${(row.cv_auc_mean * 100).toFixed(2)}%
            </td>

            <td>
                <span class="badge blue">
                    ${(row.roc_auc * 100).toFixed(2)}%
                </span>
            </td>

            <td>
                ${(row.precision * 100).toFixed(2)}%
            </td>

            <td>
                ${(row.recall * 100).toFixed(2)}%
            </td>

            <td>
                ${(row.f1_score * 100).toFixed(2)}%
            </td>

            <td>
                ${row.brier_score.toFixed(4)}
            </td>

        `;

        tbody.appendChild(tr);

    });

}


// ==========================================================================
// GLOBAL SHAP EXPLAINABILITY
// ==========================================================================

async function fetchGlobalExplainability() {

    try {

        const response =
            await fetch(
                `${API_BASE_URL}/api/explain/global`
            );

        if (!response.ok) {
            throw new Error(
                'Global explainability request failed'
            );
        }

        const data = await response.json();

        console.log(
            'Global SHAP response:',
            data
        );

        renderGlobalImportanceChart(
            data.global_feature_importance
        );

    } catch (error) {

        console.error(
            'Global SHAP error:',
            error
        );

    }

}


// ==========================================================================
// GLOBAL SHAP CHART
// ==========================================================================

function renderGlobalImportanceChart(importanceData) {

    const canvas =
        document.getElementById(
            'globalImportanceChart'
        );

    if (!canvas || !importanceData) return;

    const ctx = canvas.getContext('2d');

    const topFeatures =
        importanceData.slice(0, 8);

    const labels =
        topFeatures.map(
            feature => feature.display_name
        );

    const values =
        topFeatures.map(
            feature => feature.importance
        );

    if (globalImportanceChart) {
        globalImportanceChart.destroy();
    }

    globalImportanceChart = new Chart(ctx, {

        type: 'bar',

        data: {

            labels: labels,

            datasets: [{

                label: 'Importance Score (%)',

                data: values,

                backgroundColor:
                    'rgba(6, 182, 212, 0.75)',

                borderRadius: 6

            }]

        },

        options: {

            indexAxis: 'y',

            responsive: true,

            maintainAspectRatio: false,

            plugins: {

                legend: {
                    display: false
                }

            },

            scales: {

                x: {

                    ticks: {
                        color: '#9ca3af'
                    },

                    grid: {
                        color:
                            'rgba(255, 255, 255, 0.05)'
                    }

                },

                y: {

                    ticks: {
                        color: '#9ca3af'
                    },

                    grid: {
                        display: false
                    }

                }

            }

        }

    });

}


// ==========================================================================
// SINGLE PATIENT PREDICTION
// ==========================================================================

function initPredictorForm() {

    const form =
        document.getElementById('patient-form');

    if (!form) return;

    form.addEventListener(
        'submit',
        async event => {

            event.preventDefault();

            const payload = {

                age:
                    parseInt(
                        document.getElementById('age').value
                    ),

                sex:
                    document.getElementById('sex').value,

                chest_pain_type:
                    document.getElementById(
                        'chest_pain_type'
                    ).value,

                resting_bp:
                    parseFloat(
                        document.getElementById(
                            'resting_bp'
                        ).value
                    ),

                cholesterol:
                    parseFloat(
                        document.getElementById(
                            'cholesterol'
                        ).value
                    ),

                fasting_bs:
                    parseFloat(
                        document.getElementById(
                            'fasting_bs'
                        ).value
                    ),

                resting_ecg:
                    document.getElementById(
                        'resting_ecg'
                    ).value,

                max_hr:
                    parseFloat(
                        document.getElementById(
                            'max_hr'
                        ).value
                    ),

                exercise_angina:
                    document.getElementById(
                        'exercise_angina'
                    ).value,

                oldpeak:
                    parseFloat(
                        document.getElementById(
                            'oldpeak'
                        ).value
                    ),

                st_slope:
                    document.getElementById(
                        'st_slope'
                    ).value,

                bmi:
                    parseFloat(
                        document.getElementById(
                            'bmi'
                        ).value
                    ),

                hba1c:
                    parseFloat(
                        document.getElementById(
                            'hba1c'
                        ).value
                    )

            };


            try {

                const response =
                    await fetch(
                        `${API_BASE_URL}/api/predict`,
                        {

                            method: 'POST',

                            headers: {
                                'Content-Type':
                                    'application/json'
                            },

                            body:
                                JSON.stringify(payload)

                        }
                    );

                if (!response.ok) {

                    const errorText =
                        await response.text();

                    throw new Error(
                        `Prediction failed: ${errorText}`
                    );

                }

                const data =
                    await response.json();

                console.log(
                    'Prediction response:',
                    data
                );

                if (!data.assessment) {
                    throw new Error(
                        'Invalid prediction response'
                    );
                }

                displayPredictionResults(
                    data.assessment
                );


            } catch (error) {

                console.error(
                    'Prediction API error:',
                    error
                );

                alert(
                    'Prediction API is unavailable. Please check the backend.'
                );

            }

        }
    );

}


// ==========================================================================
// DISPLAY PATIENT PREDICTION
// ==========================================================================

function displayPredictionResults(assessment) {

    if (!assessment) return;

    const probability =
        Number(
            assessment.risk_probability
        );

    const percentElement =
        document.getElementById(
            'risk-percent-text'
        );

    const badge =
        document.getElementById(
            'risk-category-badge'
        );

    const gauge =
        document.getElementById(
            'gauge-fill'
        );


    if (percentElement) {

        percentElement.textContent =
            `${assessment.risk_percentage}%`;

    }


    if (badge) {

        badge.textContent =
            assessment.risk_category;

        badge.style.borderColor =
            assessment.risk_color;

        badge.style.color =
            assessment.risk_color;

    }


    if (gauge) {

        const safeProbability =
            Math.max(
                0,
                Math.min(
                    1,
                    probability
                )
            );

        const rotation =
            (
                safeProbability * 0.50
            ).toFixed(2);

        gauge.style.transform =
            `rotate(${rotation}turn)`;

    }


    const driversContainer =
        document.getElementById(
            'drivers-list'
        );

    if (!driversContainer) return;

    driversContainer.innerHTML = '';


    if (
        !assessment.top_drivers ||
        assessment.top_drivers.length === 0
    ) {

        driversContainer.innerHTML =
            '<div class="placeholder-text">No SHAP drivers available.</div>';

        return;

    }


    assessment.top_drivers.forEach(driver => {

        const item =
            document.createElement('div');

        item.className =
            'driver-item';

        const impact =
            Number(driver.impact) || 0;

        const isIncrease =
            impact > 0;

        const impactClass =
            isIncrease
                ? 'increase'
                : 'decrease';

        const sign =
            isIncrease
                ? '+'
                : '';

        item.innerHTML = `

            <div class="driver-info">

                <span class="driver-name">
                    ${driver.display_name}
                </span>

                <span class="driver-val">
                    Patient Value:
                    <strong>
                        ${driver.value}
                    </strong>
                </span>

            </div>

            <div class="driver-impact ${impactClass}">
                ${sign}${(impact * 100).toFixed(1)}% Risk
            </div>

        `;

        driversContainer.appendChild(item);

    });

}


// ==========================================================================
// BATCH CSV UPLOAD
// ==========================================================================
function initBatchUpload() {

    const dropzone =
        document.getElementById('dropzone');

    const fileInput =
        document.getElementById(
            'csv-file-input'
        );

    const selectButton =
        document.getElementById(
            'btn-select-file'
        );

    if (!dropzone || !fileInput || !selectButton) {
        return;
    }


    selectButton.addEventListener(
        'click',
        event => {

            event.preventDefault();

            fileInput.click();

        }
    );


    fileInput.addEventListener(
        'change',
        event => {

            if (
                event.target.files &&
                event.target.files.length > 0
            ) {

                processBatchFile(
                    event.target.files[0]
                );

            }

        }
    );


    dropzone.addEventListener(
        'dragover',
        event => {

            event.preventDefault();

        }
    );


    dropzone.addEventListener(
        'drop',
        event => {

            event.preventDefault();

            if (
                event.dataTransfer.files &&
                event.dataTransfer.files.length > 0
            ) {

                processBatchFile(
                    event.dataTransfer.files[0]
                );

            }

        }
    );

}


// ==========================================================================
// PROCESS BATCH CSV
// ==========================================================================

async function processBatchFile(file) {

    if (
        !file.name.toLowerCase().endsWith('.csv')
    ) {

        alert(
            'Please select a CSV file.'
        );

        return;

    }


    const formData =
        new FormData();

    formData.append(
        'file',
        file
    );


    try {

        const response =
            await fetch(
                `${API_BASE_URL}/api/predict-batch`,
                {

                    method: 'POST',

                    body: formData

                }
            );


        if (!response.ok) {

            const errorText =
                await response.text();

            throw new Error(
                `Batch API error: ${errorText}`
            );

        }


        const data =
            await response.json();

        console.log(
            'Batch prediction response:',
            data
        );


        renderBatchResults(data);


    } catch (error) {

        console.error(
            'Batch prediction error:',
            error
        );

        alert(
            'Batch prediction failed. Please check that your CSV contains all required columns.'
        );

    }

}


// ==========================================================================
// RENDER BATCH RESULTS
// ==========================================================================

function renderBatchResults(batchData) {

    const container =
        document.getElementById(
            'batch-results-container'
        );

    if (container) {

        container.classList.remove(
            'display-none'
        );

    }


    const summary =
        batchData.summary || {};

    const results =
        batchData.results || [];


    const total =
        batchData.total_records ??
        results.length;


    const low =
        summary.low_risk ?? 0;

    const moderate =
        summary.moderate_risk ?? 0;

    const high =
        summary.high_risk ?? 0;


    const totalElement =
        document.getElementById(
            'batch-total'
        );

    const lowElement =
        document.getElementById(
            'batch-low'
        );

    const moderateElement =
        document.getElementById(
            'batch-mod'
        );

    const highElement =
        document.getElementById(
            'batch-high'
        );


    if (totalElement)
        totalElement.textContent =
            total;

    if (lowElement)
        lowElement.textContent =
            low;

    if (moderateElement)
        moderateElement.textContent =
            moderate;

    if (highElement)
        highElement.textContent =
            high;


    const tbody =
        document.getElementById(
            'batch-table-body'
        );

    if (!tbody) return;

    tbody.innerHTML = '';


    results
        .slice(0, 25)
        .forEach(row => {

            const tr =
                document.createElement('tr');

            const percentage =
                Number(
                    row.risk_percentage
                ) || 0;


            const patientId =
                row.patient_id ??
                row.row ??
                '';


            const flag =
                percentage > 60
                    ? 'ALERT'
                    : 'NORMAL';


            const flagClass =
                percentage > 60
                    ? 'red'
                    : 'green';


            tr.innerHTML = `

                <td>
                    #PAT-${String(
                        patientId
                    ).padStart(4, '0')}
                </td>

                <td>
                    ${row.age ?? '-'}
                </td>

                <td>
                    ${row.sex ?? '-'}
                </td>

                <td>
                    <strong>
                        ${percentage.toFixed(2)}%
                    </strong>
                </td>

                <td>
                    ${row.risk_category ??
                        row.risk_level ??
                        '-'}
                </td>

                <td>
                    <span class="badge ${flagClass}">
                        ${flag}
                    </span>
                </td>

            `;

            tbody.appendChild(tr);

        });

}


// ==========================================================================
// DRIFT MONITOR
// ==========================================================================

function initDriftMonitorControls() {

    // Look for possible drift upload controls.
    // This is written defensively because your current
    // index.html portion containing the Drift tab was not visible.

    const possibleInputIds = [
        'drift-file-input',
        'drift-csv-file-input',
        'drift-file'
    ];

    const possibleButtonIds = [
        'btn-select-drift-file',
        'btn-run-drift',
        'btn-run-drift-normal'
    ];


    let input = null;

    for (const id of possibleInputIds) {

        const element =
            document.getElementById(id);

        if (element) {

            input = element;
            break;

        }

    }


    if (input) {

        input.addEventListener(
            'change',
            event => {

                if (
                    event.target.files &&
                    event.target.files.length > 0
                ) {

                    selectedDriftFile =
                        event.target.files[0];

                    console.log(
                        'Drift CSV selected:',
                        selectedDriftFile.name
                    );

                }

            }
        );

    }


    for (const id of possibleButtonIds) {

        const button =
            document.getElementById(id);

        if (!button) continue;


        button.addEventListener(
            'click',
            event => {

                event.preventDefault();

                if (
                    selectedDriftFile
                ) {

                    fetchDriftReport(
                        selectedDriftFile
                    );

                } else {

                    alert(
                        'Please select a drift CSV file first.'
                    );

                }

            }
        );

    }

}


// ==========================================================================
// DRIFT API REQUEST
// ==========================================================================

async function fetchDriftReport(file) {

    if (!file) {

        alert(
            'Please select a CSV file.'
        );

        return;

    }


    if (
        !file.name.toLowerCase().endsWith('.csv')
    ) {

        alert(
            'Drift monitoring requires a CSV file.'
        );

        return;

    }


    const formData =
        new FormData();

    formData.append(
        'file',
        file
    );


    try {

        console.log(
            'Uploading drift CSV:',
            file.name
        );


        const response =
            await fetch(
                `${API_BASE_URL}/api/drift`,
                {

                    method: 'POST',

                    body: formData

                }
            );


        if (!response.ok) {

            const errorText =
                await response.text();

            throw new Error(
                `Drift API error: ${errorText}`
            );

        }


        const data =
            await response.json();

        console.log(
            'Drift API response:',
            data
        );


        const analysis =
            data.drift_analysis ||
            data;


        renderDriftReport(
            analysis
        );


    } catch (error) {

        console.error(
            'Drift monitoring error:',
            error
        );

        alert(
            'Drift analysis failed. Please select a valid CSV file.'
        );

    }

}


// ==========================================================================
// RENDER DRIFT REPORT
// ==========================================================================

function renderDriftReport(driftData) {

    if (!driftData) return;


    const status =
        driftData.overall_status ||
        'Unknown';


    const color =
        driftData.overall_color ||
        '#9ca3af';


    const banner =
        document.getElementById(
            'drift-status-banner'
        );


    if (banner) {

        banner.textContent =
            status;

        banner.style.color =
            color;

    }


    // Update Overview KPI
    const kpiDrift =
        document.getElementById(
            'kpi-drift'
        );


    const kpiDriftSub =
        document.getElementById(
            'kpi-drift-sub'
        );


    if (kpiDrift) {

        if (
            status.includes(
                'DRIFT DETECTED'
            )
        ) {

            kpiDrift.textContent =
                'Drift Detected';

        } else {

            kpiDrift.textContent =
                'Stable';

        }

        kpiDrift.style.color =
            color;

    }


    if (kpiDriftSub) {

        if (
            driftData.feature_metrics &&
            driftData.feature_metrics.length > 0
        ) {

            const maxPSI =
                Math.max(
                    ...driftData.feature_metrics.map(
                        feature =>
                            Number(
                                feature.psi_score
                            ) || 0
                    )
                );

            kpiDriftSub.textContent =
                `Maximum PSI: ${maxPSI.toFixed(2)}`;

        }

        kpiDriftSub.style.color =
            color;

    }


    const tbody =
        document.getElementById(
            'drift-table-body'
        );


    if (!tbody) return;


    tbody.innerHTML = '';


    const metrics =
        driftData.feature_metrics || [];


    metrics.forEach(feature => {

        const tr =
            document.createElement('tr');


        tr.innerHTML = `

            <td>
                <strong>
                    ${feature.display_name ??
                        feature.feature ??
                        '-'}
                </strong>
            </td>

            <td>
                ${feature.baseline_mean ?? '-'}
            </td>

            <td>
                ${feature.current_mean ?? '-'}
            </td>

            <td>
                ${feature.psi_score ?? '-'}
            </td>

            <td>
                ${feature.ks_statistic ?? '-'}
            </td>

            <td>
                ${feature.p_value ?? '-'}
            </td>

            <td>
                <span
                    class="badge"
                    style="
                        color: ${feature.status_color ?? '#9ca3af'};
                        border: 1px solid ${feature.status_color ?? '#9ca3af'};
                    "
                >
                    ${feature.status ?? '-'}
                </span>
            </td>

        `;

        tbody.appendChild(tr);

    });

}


// ==========================================================================
// FALLBACK BENCHMARK DATA
// ==========================================================================

function renderFallbackMetrics() {

    const fallbackSummary = [

        {
            model: 'RandomForest',
            cv_auc_mean: 0.9482,
            roc_auc: 0.9520,
            precision: 0.912,
            recall: 0.925,
            f1_score: 0.918,
            brier_score: 0.0812
        },

        {
            model: 'GradientBoosting',
            cv_auc_mean: 0.9410,
            roc_auc: 0.9465,
            precision: 0.898,
            recall: 0.915,
            f1_score: 0.906,
            brier_score: 0.0890
        },

        {
            model: 'NeuralNetwork',
            cv_auc_mean: 0.9280,
            roc_auc: 0.9320,
            precision: 0.885,
            recall: 0.892,
            f1_score: 0.888,
            brier_score: 0.0980
        },

        {
            model: 'LogisticRegression',
            cv_auc_mean: 0.9150,
            roc_auc: 0.9190,
            precision: 0.870,
            recall: 0.865,
            f1_score: 0.867,
            brier_score: 0.1120
        },

        {
            model: 'SVC',
            cv_auc_mean: 0.9120,
            roc_auc: 0.9140,
            precision: 0.862,
            recall: 0.870,
            f1_score: 0.866,
            brier_score: 0.1180
        }

    ];


    renderModelComparisonChart(
        fallbackSummary
    );

    renderBenchmarkTable(
        fallbackSummary
    );

}


// ==========================================================================
// END OF MEDPULSE AI FRONTEND
// ==========================================================================
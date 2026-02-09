// ========================================
// Next-Gen CAPTCHAs - Main JavaScript
// ========================================

// CAPTCHA Data - 27 types with metadata from the paper
const captchaData = [
    {
        id: "3D_Viewpoint",
        name: "3D Viewpoint",
        description: "Select all views showing the same colored-edge wireframe from different angles",
        gaps: ["G1", "G4", "G5"],
        answerType: "Select",
        generative: true
    },
    {
        id: "Backmost_Layer",
        name: "Backmost Layer",
        description: "Click cells where the backmost (occluded) shape matches the reference",
        gaps: ["G1"],
        answerType: "Select",
        generative: true
    },
    {
        id: "Box_Folding",
        name: "Box Folding",
        description: "Choose the folded cube that matches the given 2D net",
        gaps: ["G1", "G4"],
        answerType: "Select",
        generative: true
    },
    {
        id: "Color_Counting",
        name: "Color Counting",
        description: "Select cells that meet the rule about the number of colors in each cell",
        gaps: ["G3"],
        answerType: "Select",
        generative: true
    },
    {
        id: "Dice_Roll_Path",
        name: "Dice Roll Path",
        description: "Roll a die along a shown path and report the final top face",
        gaps: ["G3", "G4", "G5"],
        answerType: "Numeric",
        generative: true
    },
    {
        id: "Dynamic_Jigsaw",
        name: "Dynamic Jigsaw",
        description: "Drag and drop animated GIF pieces to complete a 3×3 jigsaw puzzle",
        gaps: ["G2", "G4", "G5"],
        answerType: "Drag-and-drop",
        generative: true
    },
    {
        id: "Hole_Counting",
        name: "Hole Counting",
        description: "Count topological holes in presented glyphs and shapes",
        gaps: ["G1", "G3"],
        answerType: "Numeric",
        generative: true
    },
    {
        id: "Illusory_Ribbons",
        name: "Illusory Ribbons",
        description: "Select cells containing exactly the target number of ribbon loops",
        gaps: ["G1", "G3"],
        answerType: "Select",
        generative: true
    },
    {
        id: "Layered_Stack",
        name: "Layered Stack",
        description: "Select cells where top shape and counts in lower layers meet a rule",
        gaps: ["G1", "G3"],
        answerType: "Select",
        generative: true
    },
    {
        id: "Mirror",
        name: "Mirror",
        description: "Find mirror options that do not match the reflected reference",
        gaps: ["G1"],
        answerType: "Select",
        generative: false
    },
    {
        id: "Multi_Script",
        name: "Multi Script",
        description: "Select cells containing any target characters across multiple writing systems",
        gaps: ["G1"],
        answerType: "Select",
        generative: true
    },
    {
        id: "Occluded_Pattern_Counting",
        name: "Occluded Pattern Counting",
        description: "Count two specified shapes under a semi-transparent occluder",
        gaps: ["G1", "G3"],
        answerType: "Numeric",
        generative: true
    },
    {
        id: "Red_Dot",
        name: "Red Dot",
        description: "Timed clicks on appearing red dots until hit quota - tests reaction speed",
        gaps: ["G5"],
        answerType: "Click position",
        generative: true
    },
    {
        id: "Rotation_Match",
        name: "Rotation Match",
        description: "Select tiles of the most frequent shape, ignoring rotation/color/texture",
        gaps: ["G1", "G4"],
        answerType: "Select",
        generative: true
    },
    {
        id: "Shadow_Direction",
        name: "Shadow Direction",
        description: "Match light-source direction from photorealistic 3D shadows",
        gaps: ["G1"],
        answerType: "Select",
        generative: true
    },
    {
        id: "Shadow_Plausible",
        name: "Shadow Plausible",
        description: "Pick images with physically plausible shadows in a grid",
        gaps: ["G1"],
        answerType: "Select",
        generative: false
    },
    {
        id: "Spooky_Circle",
        name: "Spooky Circle",
        description: "Count circles only visible via motion-contrast noise",
        gaps: ["G2"],
        answerType: "Numeric",
        generative: true
    },
    {
        id: "Spooky_Circle_Grid",
        name: "Spooky Circle Grid",
        description: "Count how many grid cells contain motion-contrast circles",
        gaps: ["G2", "G3"],
        answerType: "Numeric",
        generative: true
    },
    {
        id: "Spooky_Jigsaw",
        name: "Spooky Jigsaw",
        description: "Drag and drop motion-contrast pieces to complete the jigsaw",
        gaps: ["G2", "G4", "G5"],
        answerType: "Drag-and-drop",
        generative: true
    },
    {
        id: "Spooky_Shape_Grid",
        name: "Spooky Shape Grid",
        description: "Select spooky cells with the target shape and rotation direction",
        gaps: ["G2"],
        answerType: "Select",
        generative: true
    },
    {
        id: "Spooky_Size",
        name: "Spooky Size",
        description: "Click largest/smallest target shape visible only via motion contrast",
        gaps: ["G2", "G5"],
        answerType: "Click position",
        generative: true
    },
    {
        id: "Spooky_Text",
        name: "Spooky Text",
        description: "Read and type text visible only via motion contrast",
        gaps: ["G2"],
        answerType: "Text entry",
        generative: true
    },
    {
        id: "Static_Jigsaw",
        name: "Static Jigsaw",
        description: "Drag and drop static pieces to complete a jigsaw puzzle",
        gaps: ["G4", "G5"],
        answerType: "Drag-and-drop",
        generative: true
    },
    {
        id: "Structure_From_Motion",
        name: "Structure From Motion",
        description: "Select GIF cells whose dot motion reflects the same rigid 3D shape",
        gaps: ["G2"],
        answerType: "Select",
        generative: true
    },
    {
        id: "Subway_Paths",
        name: "Subway Paths",
        description: "Select maps with the specified count of valid routes under stamp rules",
        gaps: ["G3", "G4"],
        answerType: "Select",
        generative: true
    },
    {
        id: "Temporal_Object_Continuity",
        name: "Temporal Object Continuity",
        description: "Select GIF cells where identity changes behind occluders",
        gaps: ["G2", "G4"],
        answerType: "Select",
        generative: true
    },
    {
        id: "Trajectory_Recovery",
        name: "Trajectory Recovery",
        description: "Watch a reference trajectory GIF; select matching trajectory plots",
        gaps: ["G2", "G4"],
        answerType: "Select",
        generative: true
    }
];

// Results data from Table 2 in the paper
const resultsData = [
    { type: "3D_Viewpoint", human: 100.0, gpt52: 0.0, gemini3flash: 0.0, claudeopus: 0.0, gemini3pro: 0.0, doubao: 0.0, qwen3: 0.0 },
    { type: "Backmost_Layer", human: 100.0, gpt52: 20.0, gemini3flash: 0.0, claudeopus: 0.0, gemini3pro: 0.0, doubao: 0.0, qwen3: 0.0 },
    { type: "Box_Folding", human: 100.0, gpt52: 20.0, gemini3flash: 0.0, claudeopus: 0.0, gemini3pro: 0.0, doubao: 0.0, qwen3: 0.0 },
    { type: "Color_Counting", human: 100.0, gpt52: 40.0, gemini3flash: 5.0, claudeopus: 0.0, gemini3pro: 0.0, doubao: 0.0, qwen3: 5.0 },
    { type: "Dice_Roll_Path", human: 100.0, gpt52: 0.0, gemini3flash: 15.0, claudeopus: 0.0, gemini3pro: 0.0, doubao: 5.0, qwen3: 15.0 },
    { type: "Dynamic_Jigsaw", human: 100.0, gpt52: 0.0, gemini3flash: 0.0, claudeopus: 0.0, gemini3pro: 0.0, doubao: 0.0, qwen3: 0.0 },
    { type: "Hole_Counting", human: 100.0, gpt52: 0.0, gemini3flash: 0.0, claudeopus: 0.0, gemini3pro: 0.0, doubao: 0.0, qwen3: 0.0 },
    { type: "Illusory_Ribbons", human: 100.0, gpt52: 0.0, gemini3flash: 0.0, claudeopus: 0.0, gemini3pro: 0.0, doubao: 0.0, qwen3: 0.0 },
    { type: "Layered_Stack", human: 100.0, gpt52: 0.0, gemini3flash: 10.0, claudeopus: 0.0, gemini3pro: 0.0, doubao: 0.0, qwen3: 0.0 },
    { type: "Mirror", human: 90.0, gpt52: 20.0, gemini3flash: 18.2, claudeopus: 0.0, gemini3pro: 9.1, doubao: 9.1, qwen3: 0.0 },
    { type: "Multi_Script", human: 100.0, gpt52: 0.0, gemini3flash: 0.0, claudeopus: 0.0, gemini3pro: 0.0, doubao: 0.0, qwen3: 0.0 },
    { type: "Occluded_Pattern_Counting", human: 100.0, gpt52: 20.0, gemini3flash: 0.0, claudeopus: 0.0, gemini3pro: 5.0, doubao: 15.0, qwen3: 0.0 },
    { type: "Red_Dot", human: 100.0, gpt52: 0.0, gemini3flash: 15.0, claudeopus: 20.0, gemini3pro: 0.0, doubao: 0.0, qwen3: 0.0 },
    { type: "Rotation_Match", human: 100.0, gpt52: 20.0, gemini3flash: 0.0, claudeopus: 0.0, gemini3pro: 0.0, doubao: 0.0, qwen3: 0.0 },
    { type: "Shadow_Direction", human: 100.0, gpt52: 20.0, gemini3flash: 0.0, claudeopus: 0.0, gemini3pro: 0.0, doubao: 0.0, qwen3: 0.0 },
    { type: "Shadow_Plausible", human: 100.0, gpt52: 0.0, gemini3flash: 12.5, claudeopus: 0.0, gemini3pro: 0.0, doubao: 0.0, qwen3: 0.0 },
    { type: "Spooky_Circle", human: 100.0, gpt52: 0.0, gemini3flash: 5.0, claudeopus: 0.0, gemini3pro: 0.0, doubao: 5.0, qwen3: 0.0 },
    { type: "Spooky_Circle_Grid", human: 100.0, gpt52: 0.0, gemini3flash: 0.0, claudeopus: 0.0, gemini3pro: 5.0, doubao: 0.0, qwen3: 0.0 },
    { type: "Spooky_Jigsaw", human: 90.0, gpt52: 0.0, gemini3flash: 0.0, claudeopus: 0.0, gemini3pro: 0.0, doubao: 0.0, qwen3: 0.0 },
    { type: "Spooky_Shape_Grid", human: 100.0, gpt52: 0.0, gemini3flash: 0.0, claudeopus: 0.0, gemini3pro: 0.0, doubao: 0.0, qwen3: 0.0 },
    { type: "Spooky_Size", human: 100.0, gpt52: 0.0, gemini3flash: 5.0, claudeopus: 0.0, gemini3pro: 10.0, doubao: 0.0, qwen3: 5.0 },
    { type: "Spooky_Text", human: 100.0, gpt52: 0.0, gemini3flash: 0.0, claudeopus: 0.0, gemini3pro: 0.0, doubao: 0.0, qwen3: 0.0 },
    { type: "Static_Jigsaw", human: 100.0, gpt52: 0.0, gemini3flash: 0.0, claudeopus: 60.0, gemini3pro: 0.0, doubao: 0.0, qwen3: 0.0 },
    { type: "Structure_From_Motion", human: 90.0, gpt52: 0.0, gemini3flash: 0.0, claudeopus: 0.0, gemini3pro: 5.0, doubao: 0.0, qwen3: 0.0 },
    { type: "Subway_Paths", human: 100.0, gpt52: 0.0, gemini3flash: 0.0, claudeopus: 0.0, gemini3pro: 0.0, doubao: 0.0, qwen3: 0.0 },
    { type: "Temporal_Object_Continuity", human: 100.0, gpt52: 0.0, gemini3flash: 0.0, claudeopus: 0.0, gemini3pro: 0.0, doubao: 0.0, qwen3: 0.0 },
    { type: "Trajectory_Recovery", human: 100.0, gpt52: 0.0, gemini3flash: 0.0, claudeopus: 0.0, gemini3pro: 0.0, doubao: 0.0, qwen3: 0.0 }
];

// Gap category names for display
const gapNames = {
    "G1": "Scene Structure",
    "G2": "Temporal",
    "G3": "Numerosity",
    "G4": "State Tracking",
    "G5": "Action"
};

// Gap colors
const gapColors = {
    "G1": "#3b82f6",
    "G2": "#8b5cf6",
    "G3": "#ec4899",
    "G4": "#f59e0b",
    "G5": "#10b981"
};

// ========================================
// Initialize on DOM ready
// ========================================
document.addEventListener('DOMContentLoaded', function() {
    renderCaptchaGallery();
    renderResultsTable();
    setupFilterButtons();
});

// ========================================
// Render CAPTCHA Gallery
// ========================================
function renderCaptchaGallery() {
    const grid = document.getElementById('captcha-grid');
    if (!grid) return;

    grid.innerHTML = captchaData.map((captcha, index) => {
        const gapTags = captcha.gaps.map(g =>
            `<span class="captcha-tag gap-tag" style="background: ${gapColors[g]}">${g}</span>`
        ).join('');

        // Use first GIF for preview (index 0)
        const gifPath = `assets/gifs/${captcha.id}_0.gif`;

        return `
            <div class="captcha-card" data-gaps="${captcha.gaps.join(',')}" style="animation-delay: ${index * 0.05}s">
                <div class="captcha-preview">
                    <img src="${gifPath}" alt="${captcha.name}" loading="lazy">
                </div>
                <div class="captcha-info">
                    <div class="captcha-name">${captcha.name}</div>
                    <div class="captcha-desc">${captcha.description}</div>
                    <div class="captcha-tags">
                        ${gapTags}
                        <span class="captcha-tag">${captcha.answerType}</span>
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

// ========================================
// Render Results Table
// ========================================
function renderResultsTable() {
    const tbody = document.getElementById('results-tbody');
    if (!tbody) return;

    // Helper to get cell class based on value
    function getCellClass(value) {
        if (value >= 90) return 'cell-high';
        if (value >= 20) return 'cell-medium';
        if (value > 0) return 'cell-low';
        return 'cell-zero';
    }

    // Format value for display
    function formatValue(value) {
        if (value === 0) return '0.0';
        return value.toFixed(1);
    }

    // Add average row first
    const avgRow = {
        type: "Average",
        human: 98.8,
        gpt52: 5.9,
        gemini3flash: 3.2,
        claudeopus: 3.0,
        gemini3pro: 1.3,
        doubao: 1.3,
        qwen3: 0.9
    };

    let html = `
        <tr style="font-weight: 600; background: rgba(0, 50, 98, 0.05);">
            <td>Avg Pass@1 (%)</td>
            <td class="${getCellClass(avgRow.human)}">${formatValue(avgRow.human)}</td>
            <td class="${getCellClass(avgRow.gpt52)}">${formatValue(avgRow.gpt52)}</td>
            <td class="${getCellClass(avgRow.gemini3flash)}">${formatValue(avgRow.gemini3flash)}</td>
            <td class="${getCellClass(avgRow.claudeopus)}">${formatValue(avgRow.claudeopus)}</td>
            <td class="${getCellClass(avgRow.gemini3pro)}">${formatValue(avgRow.gemini3pro)}</td>
            <td class="${getCellClass(avgRow.doubao)}">${formatValue(avgRow.doubao)}</td>
            <td class="${getCellClass(avgRow.qwen3)}">${formatValue(avgRow.qwen3)}</td>
        </tr>
    `;

    // Add individual rows
    html += resultsData.map(row => {
        const captcha = captchaData.find(c => c.id === row.type);
        const displayName = captcha ? captcha.name : row.type.replace(/_/g, ' ');

        return `
            <tr>
                <td>${displayName}</td>
                <td class="${getCellClass(row.human)}">${formatValue(row.human)}</td>
                <td class="${getCellClass(row.gpt52)}">${formatValue(row.gpt52)}</td>
                <td class="${getCellClass(row.gemini3flash)}">${formatValue(row.gemini3flash)}</td>
                <td class="${getCellClass(row.claudeopus)}">${formatValue(row.claudeopus)}</td>
                <td class="${getCellClass(row.gemini3pro)}">${formatValue(row.gemini3pro)}</td>
                <td class="${getCellClass(row.doubao)}">${formatValue(row.doubao)}</td>
                <td class="${getCellClass(row.qwen3)}">${formatValue(row.qwen3)}</td>
            </tr>
        `;
    }).join('');

    tbody.innerHTML = html;
}

// ========================================
// Filter Functionality
// ========================================
function setupFilterButtons() {
    const filterBtns = document.querySelectorAll('.filter-btn');
    const cards = document.querySelectorAll('.captcha-card');

    filterBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            // Update active button
            filterBtns.forEach(b => b.classList.remove('active'));
            this.classList.add('active');

            const filter = this.dataset.filter;

            // Filter cards
            cards.forEach(card => {
                if (filter === 'all') {
                    card.classList.remove('hidden');
                } else {
                    const cardGaps = card.dataset.gaps.split(',');
                    if (cardGaps.includes(filter)) {
                        card.classList.remove('hidden');
                    } else {
                        card.classList.add('hidden');
                    }
                }
            });
        });
    });
}

// ========================================
// Copy BibTeX
// ========================================
function copyBibtex() {
    const bibtex = `@article{liu2026nextgen,
  title={Next-Gen CAPTCHAs: Leveraging the Cognitive Gap for
         Scalable and Diverse GUI-Agent Defense},
  author={Liu, Jiacheng and Luo, Yaxin and Cui, Jiacheng and
          Shang, Xinyi and Zhao, Xiaohan and Shen, Zhiqiang},
  year={2026}
}`;

    navigator.clipboard.writeText(bibtex).then(() => {
        const btn = document.querySelector('.copy-btn');
        const originalText = btn.innerHTML;
        btn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg> Copied!';
        setTimeout(() => {
            btn.innerHTML = originalText;
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy:', err);
    });
}

// ========================================
// Smooth scroll for anchor links
// ========================================
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

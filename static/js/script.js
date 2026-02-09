let puzzleStartTime = null;
let actionSequence = [];
let timerInterval = null;

// Function to log user actions
function logAction(actionType, details = {}) {
    const timestamp = Date.now();
    const timeSinceStart = puzzleStartTime ? timestamp - puzzleStartTime : 0;
    actionSequence.push({
        type: actionType,
        timestamp: timestamp,
        timeSinceStart: timeSinceStart,
        details: details
    });
    console.log('Action logged:', actionType, details);
}

// Function to update elapsed time display
function updateElapsedTime() {
    const elapsedTimeEl = document.getElementById('elapsed-time');
    if (puzzleStartTime && elapsedTimeEl) {
        const elapsedSeconds = ((Date.now() - puzzleStartTime) / 1000).toFixed(1);
        elapsedTimeEl.textContent = `${elapsedSeconds}s`;
    }
}

// Function to start the timer
function startTimer() {
    if (timerInterval) {
        clearInterval(timerInterval);
    }
    timerInterval = setInterval(updateElapsedTime, 100); // Update every 100ms
}

// Function to stop the timer
function stopTimer() {
    if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
    }
}

// ============================================================
// Interactive 3D Dice Reference Classes for Dice_Roll_Path
// ============================================================

/**
 * DiceOrientation - Tracks the state of a standard die
 * Mirrors the Python DiceOrientation dataclass
 */
class DiceOrientation {
    constructor(top, bottom, north, south, east, west) {
        this.top = top;
        this.bottom = bottom;
        this.north = north;
        this.south = south;
        this.east = east;
        this.west = west;
    }

    /**
     * Roll the die in a direction and return new orientation
     * @param {string} direction - 'N', 'S', 'E', or 'W'
     * @returns {DiceOrientation} New orientation after roll
     */
    roll(direction) {
        const d = direction.toUpperCase();
        if (d === 'N') {
            // Rolling north: top goes to north, south goes to top
            return new DiceOrientation(
                this.south, this.north, this.top, this.bottom, this.east, this.west
            );
        }
        if (d === 'S') {
            // Rolling south: top goes to south, north goes to top
            return new DiceOrientation(
                this.north, this.south, this.bottom, this.top, this.east, this.west
            );
        }
        if (d === 'E') {
            // Rolling east: top goes to east, west goes to top
            return new DiceOrientation(
                this.west, this.east, this.north, this.south, this.top, this.bottom
            );
        }
        if (d === 'W') {
            // Rolling west: top goes to west, east goes to top
            return new DiceOrientation(
                this.east, this.west, this.north, this.south, this.bottom, this.top
            );
        }
        throw new Error(`Invalid direction: ${direction}`);
    }

    clone() {
        return new DiceOrientation(
            this.top, this.bottom, this.north, this.south, this.east, this.west
        );
    }
}

/**
 * InteractiveDiceReference - Three.js 3D dice that users can rotate
 * Uses accumulated rotations to avoid texture swapping glitches
 */
class InteractiveDiceReference {
    constructor(container) {
        this.container = container;
        // Standard dice orientation: 1 on top
        this.initialOrientation = new DiceOrientation(1, 6, 2, 5, 3, 4);
        this.currentOrientation = this.initialOrientation.clone();
        this.isAnimating = false;

        // Three.js components
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.dice = null;
        this.animationId = null;

        // Track accumulated rotation using quaternions for proper composition
        this.targetQuaternion = new THREE.Quaternion();

        this.init();
    }

    init() {
        this.createContainer();
        this.createScene();
        this.createDice();
        this.createControls();
        this.updateTopFaceDisplay();
        this.startRenderLoop();
    }

    createContainer() {
        // Main wrapper
        this.wrapper = document.createElement('div');
        this.wrapper.className = 'dice-3d-reference';

        // Title
        const title = document.createElement('div');
        title.className = 'dice-3d-title';
        title.textContent = 'Interactive Dice Simulator';
        this.wrapper.appendChild(title);

        // Subtitle
        const subtitle = document.createElement('div');
        subtitle.className = 'dice-3d-subtitle';
        subtitle.textContent = 'Click N/S/E/W to roll the dice';
        this.wrapper.appendChild(subtitle);

        // Hint about opposite faces
        const hint = document.createElement('div');
        hint.className = 'dice-3d-hint';
        hint.textContent = 'Opposite faces sum to 7 (1-6, 2-5, 3-4)';
        this.wrapper.appendChild(hint);

        // Canvas container
        this.canvasContainer = document.createElement('div');
        this.canvasContainer.className = 'dice-3d-canvas-container';
        this.wrapper.appendChild(this.canvasContainer);

        // Top face display
        this.topFaceDisplay = document.createElement('div');
        this.topFaceDisplay.className = 'dice-3d-top-display';
        this.wrapper.appendChild(this.topFaceDisplay);

        // Controls container
        this.controlsContainer = document.createElement('div');
        this.controlsContainer.className = 'dice-3d-controls';
        this.wrapper.appendChild(this.controlsContainer);

        this.container.appendChild(this.wrapper);
    }

    createScene() {
        const width = 260;
        const height = 260;

        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xf8fafc);

        // Camera - isometric-like perspective
        this.camera = new THREE.PerspectiveCamera(40, width / height, 0.1, 1000);
        this.camera.position.set(3.5, 3.5, 3.5);
        this.camera.lookAt(0, 0, 0);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.canvasContainer.appendChild(this.renderer.domElement);

        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(5, 10, 7);
        this.scene.add(directionalLight);

        // Add subtle second light for better visibility
        const fillLight = new THREE.DirectionalLight(0xffffff, 0.3);
        fillLight.position.set(-5, 5, -5);
        this.scene.add(fillLight);
    }

    createDiceFaceTexture(pips) {
        const canvas = document.createElement('canvas');
        canvas.width = 128;
        canvas.height = 128;
        const ctx = canvas.getContext('2d');

        // White background with rounded corners effect
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, 128, 128);

        // Dark border
        ctx.strokeStyle = '#374151';
        ctx.lineWidth = 4;
        ctx.strokeRect(2, 2, 124, 124);

        // Pip positions for standard dice
        const pipPositions = {
            1: [[64, 64]],
            2: [[96, 32], [32, 96]],
            3: [[96, 32], [64, 64], [32, 96]],
            4: [[32, 32], [96, 32], [32, 96], [96, 96]],
            5: [[32, 32], [96, 32], [64, 64], [32, 96], [96, 96]],
            6: [[32, 32], [32, 64], [32, 96], [96, 32], [96, 64], [96, 96]]
        };

        // Draw pips
        ctx.fillStyle = '#111827';
        const positions = pipPositions[pips] || [];
        const pipRadius = 12;

        positions.forEach(([x, y]) => {
            ctx.beginPath();
            ctx.arc(x, y, pipRadius, 0, Math.PI * 2);
            ctx.fill();
        });

        const texture = new THREE.CanvasTexture(canvas);
        texture.needsUpdate = true;
        return texture;
    }

    createDice() {
        const geometry = new THREE.BoxGeometry(1.6, 1.6, 1.6);

        // Create materials for all six faces with FIXED textures
        // Three.js BoxGeometry face order: +X, -X, +Y, -Y, +Z, -Z
        // Standard dice at rest: 1 on top (+Y), 6 on bottom (-Y)
        // North=2 (-Z), South=5 (+Z), East=3 (+X), West=4 (-X)
        this.diceMaterials = [
            new THREE.MeshLambertMaterial({ map: this.createDiceFaceTexture(3) }),  // +X = East = 3
            new THREE.MeshLambertMaterial({ map: this.createDiceFaceTexture(4) }),  // -X = West = 4
            new THREE.MeshLambertMaterial({ map: this.createDiceFaceTexture(1) }),  // +Y = Top = 1
            new THREE.MeshLambertMaterial({ map: this.createDiceFaceTexture(6) }),  // -Y = Bottom = 6
            new THREE.MeshLambertMaterial({ map: this.createDiceFaceTexture(5) }),  // +Z = South = 5
            new THREE.MeshLambertMaterial({ map: this.createDiceFaceTexture(2) })   // -Z = North = 2
        ];

        this.dice = new THREE.Mesh(geometry, this.diceMaterials);

        // Add rounded edges effect with slightly darker edges
        const edges = new THREE.EdgesGeometry(geometry);
        const edgeMaterial = new THREE.LineBasicMaterial({ color: 0x374151, linewidth: 2 });
        const edgeLines = new THREE.LineSegments(edges, edgeMaterial);
        this.dice.add(edgeLines);

        this.scene.add(this.dice);
    }

    createControls() {
        // Direction buttons layout:
        //        [N]
        //   [W]  [R]  [E]
        //        [S]

        this.controlsContainer.innerHTML = `
            <div class="dice-3d-btn-row">
                <button class="dice-3d-btn dice-3d-btn-n" data-dir="N">N</button>
            </div>
            <div class="dice-3d-btn-row">
                <button class="dice-3d-btn dice-3d-btn-w" data-dir="W">W</button>
                <button class="dice-3d-reset-btn">Reset</button>
                <button class="dice-3d-btn dice-3d-btn-e" data-dir="E">E</button>
            </div>
            <div class="dice-3d-btn-row">
                <button class="dice-3d-btn dice-3d-btn-s" data-dir="S">S</button>
            </div>
        `;

        // Add event listeners
        this.controlsContainer.querySelectorAll('.dice-3d-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const dir = btn.dataset.dir;
                if (dir) this.rollDice(dir);
            });
        });

        this.controlsContainer.querySelector('.dice-3d-reset-btn').addEventListener('click', () => {
            this.reset();
        });
    }

    rollDice(direction) {
        if (this.isAnimating) return;
        this.isAnimating = true;

        // Disable buttons during animation
        this.setButtonsEnabled(false);

        // Update logical orientation immediately
        this.currentOrientation = this.currentOrientation.roll(direction);

        // Calculate rotation quaternion for this roll
        // N/S rotates around X axis, E/W rotates around Z axis
        const axis = (direction === 'N' || direction === 'S')
            ? new THREE.Vector3(1, 0, 0)
            : new THREE.Vector3(0, 0, 1);

        // Determine rotation direction (using right-hand rule)
        // N: -X rotation (top goes to north/-Z, south comes to top)
        // S: +X rotation (top goes to south/+Z, north comes to top)
        // E: -Z rotation (top goes to east/+X, west comes to top)
        // W: +Z rotation (top goes to west/-X, east comes to top)
        const angle = (direction === 'S' || direction === 'W')
            ? Math.PI / 2
            : -Math.PI / 2;

        // Create rotation quaternion and compose with current target
        const rollQuaternion = new THREE.Quaternion().setFromAxisAngle(axis, angle);
        const newTarget = new THREE.Quaternion().multiplyQuaternions(rollQuaternion, this.targetQuaternion);

        // Store starting quaternion
        const startQuaternion = this.dice.quaternion.clone();

        // Animation parameters
        const duration = 350; // ms
        const startTime = Date.now();

        const animateRoll = () => {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / duration, 1);

            // Ease-out cubic for smooth deceleration
            const eased = 1 - Math.pow(1 - progress, 3);

            // Interpolate quaternion
            this.dice.quaternion.slerpQuaternions(startQuaternion, newTarget, eased);

            if (progress < 1) {
                requestAnimationFrame(animateRoll);
            } else {
                // Animation complete - ensure exact final position
                this.dice.quaternion.copy(newTarget);
                this.targetQuaternion.copy(newTarget);

                this.updateTopFaceDisplay();
                this.isAnimating = false;
                this.setButtonsEnabled(true);
            }
        };

        animateRoll();
    }

    setButtonsEnabled(enabled) {
        this.controlsContainer.querySelectorAll('button').forEach(btn => {
            btn.disabled = !enabled;
        });
    }

    updateTopFaceDisplay() {
        this.topFaceDisplay.innerHTML = `
            <span class="dice-3d-top-label">Top Face:</span>
            <span class="dice-3d-top-value">${this.currentOrientation.top}</span>
        `;
    }

    reset() {
        if (this.isAnimating) return;
        this.isAnimating = true;
        this.setButtonsEnabled(false);

        // Reset to initial orientation
        this.currentOrientation = this.initialOrientation.clone();

        // Animate back to identity quaternion
        const startQuaternion = this.dice.quaternion.clone();
        const identityQuaternion = new THREE.Quaternion();

        const duration = 400;
        const startTime = Date.now();

        const animateReset = () => {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3);

            this.dice.quaternion.slerpQuaternions(startQuaternion, identityQuaternion, eased);

            if (progress < 1) {
                requestAnimationFrame(animateReset);
            } else {
                this.dice.quaternion.copy(identityQuaternion);
                this.targetQuaternion.copy(identityQuaternion);
                this.updateTopFaceDisplay();
                this.isAnimating = false;
                this.setButtonsEnabled(true);
            }
        };

        animateReset();
    }

    startRenderLoop() {
        const render = () => {
            this.animationId = requestAnimationFrame(render);
            this.renderer.render(this.scene, this.camera);
        };
        render();
    }

    dispose() {
        // Stop render loop
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }

        // Dispose Three.js resources
        if (this.renderer) {
            this.renderer.dispose();
        }

        if (this.diceMaterials) {
            this.diceMaterials.forEach(mat => {
                if (mat.map) mat.map.dispose();
                mat.dispose();
            });
        }

        // Remove DOM element
        if (this.wrapper && this.wrapper.parentNode) {
            this.wrapper.parentNode.removeChild(this.wrapper);
        }
    }
}

// Global reference for cleanup
window.currentDiceReference = null;

/**
 * InteractiveViewpointReference - Three.js 3D wireframe viewer for 3D_Viewpoint CAPTCHA
 * Allows continuous rotation via Up/Down/Left/Right buttons (hold to rotate)
 */
class InteractiveViewpointReference {
    constructor(container, shapeData) {
        this.container = container;
        this.shapeData = shapeData;  // { edges: [...], n_sides, name }

        // Rotation state (elevation/azimuth like matplotlib)
        this.elevation = 20;   // degrees
        this.azimuth = -35;    // degrees

        // Continuous rotation tracking
        this.rotationInterval = null;

        // Rotation speed (degrees per frame at ~60fps)
        this.rotationSpeed = 2;

        // Three.js components
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.wireframeGroup = null;
        this.animationId = null;

        this.init();
    }

    init() {
        this.createContainer();
        this.createScene();
        this.createWireframe();
        this.createControls();
        this.updateCameraPosition();
        this.startRenderLoop();
    }

    createContainer() {
        this.wrapper = document.createElement('div');
        this.wrapper.className = 'viewpoint-3d-reference';

        // Left side: canvas
        const leftSection = document.createElement('div');
        leftSection.className = 'viewpoint-3d-left';

        this.canvasContainer = document.createElement('div');
        this.canvasContainer.className = 'viewpoint-3d-canvas-container';
        leftSection.appendChild(this.canvasContainer);

        this.wrapper.appendChild(leftSection);

        // Right side: controls
        this.controlsContainer = document.createElement('div');
        this.controlsContainer.className = 'viewpoint-3d-controls';
        this.wrapper.appendChild(this.controlsContainer);

        this.container.appendChild(this.wrapper);
    }

    createScene() {
        const width = 160;
        const height = 160;

        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xffffff);

        // Orthographic camera for consistent wireframe view
        // Match matplotlib's xlim/ylim of [-1.5, 1.5] (range of 3)
        const aspect = width / height;
        const frustumSize = 3;
        this.camera = new THREE.OrthographicCamera(
            -frustumSize * aspect / 2,
            frustumSize * aspect / 2,
            frustumSize / 2,
            -frustumSize / 2,
            0.1, 1000
        );

        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(width, height);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.canvasContainer.appendChild(this.renderer.domElement);
    }

    createWireframe() {
        this.wireframeGroup = new THREE.Group();

        // Use cylinders for thick edges (WebGL ignores linewidth)
        const edgeRadius = 0.025;  // Thickness of edges

        for (const edge of this.shapeData.edges) {
            // Matplotlib uses z-up, Three.js uses y-up: swap y and z
            // Matplotlib's box_aspect([1,1,1]) with zlim=[-1,1] (range 2) vs xlim/ylim=[-1.5,1.5] (range 3)
            // stretches z by 3/2 = 1.5x to fit the same physical box
            const zScale = 1.5;
            const start = new THREE.Vector3(
                edge.start[0], edge.start[2] * zScale, -edge.start[1]
            );
            const end = new THREE.Vector3(
                edge.end[0], edge.end[2] * zScale, -edge.end[1]
            );

            // Calculate edge length and midpoint
            const direction = new THREE.Vector3().subVectors(end, start);
            const length = direction.length();
            const midpoint = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);

            // Create cylinder geometry
            const geometry = new THREE.CylinderGeometry(edgeRadius, edgeRadius, length, 8);
            const material = new THREE.MeshBasicMaterial({ color: edge.color });
            const cylinder = new THREE.Mesh(geometry, material);

            // Position at midpoint
            cylinder.position.copy(midpoint);

            // Orient cylinder along edge direction
            cylinder.quaternion.setFromUnitVectors(
                new THREE.Vector3(0, 1, 0),
                direction.normalize()
            );

            this.wireframeGroup.add(cylinder);
        }

        this.scene.add(this.wireframeGroup);
    }

    createControls() {
        // Vertical D-pad on right side:
        //   [Up]
        //  [L][R]
        //  [Down]
        // [Reset]

        this.controlsContainer.innerHTML = `
            <button class="viewpoint-3d-btn" data-dir="up">&#9650;</button>
            <div class="viewpoint-3d-btn-row">
                <button class="viewpoint-3d-btn viewpoint-3d-btn-sm" data-dir="left">&#9664;</button>
                <button class="viewpoint-3d-btn viewpoint-3d-btn-sm" data-dir="right">&#9654;</button>
            </div>
            <button class="viewpoint-3d-btn" data-dir="down">&#9660;</button>
            <button class="viewpoint-3d-reset-btn">&#x21BA;</button>
        `;

        // Continuous rotation on hold (mousedown/mouseup, touchstart/touchend)
        this.controlsContainer.querySelectorAll('.viewpoint-3d-btn').forEach(btn => {
            const startRotation = (e) => {
                e.preventDefault();
                const dir = btn.dataset.dir;
                this.startContinuousRotation(dir);
            };

            const stopRotation = (e) => {
                e.preventDefault();
                this.stopContinuousRotation();
            };

            // Mouse events
            btn.addEventListener('mousedown', startRotation);
            btn.addEventListener('mouseup', stopRotation);
            btn.addEventListener('mouseleave', stopRotation);

            // Touch events for mobile
            btn.addEventListener('touchstart', startRotation);
            btn.addEventListener('touchend', stopRotation);
            btn.addEventListener('touchcancel', stopRotation);
        });

        // Reset button
        this.controlsContainer.querySelector('.viewpoint-3d-reset-btn').addEventListener('click', () => {
            this.reset();
        });
    }

    startContinuousRotation(direction) {
        if (this.rotationInterval) return;

        this.rotationInterval = setInterval(() => {
            this.rotateStep(direction);
        }, 16);  // ~60fps
    }

    stopContinuousRotation() {
        if (this.rotationInterval) {
            clearInterval(this.rotationInterval);
            this.rotationInterval = null;
        }
    }

    rotateStep(direction) {
        switch (direction) {
            case 'up':
                this.elevation = Math.min(90, this.elevation + this.rotationSpeed);
                break;
            case 'down':
                this.elevation = Math.max(-90, this.elevation - this.rotationSpeed);
                break;
            case 'left':
                this.azimuth = (this.azimuth - this.rotationSpeed + 360) % 360;
                break;
            case 'right':
                this.azimuth = (this.azimuth + this.rotationSpeed) % 360;
                break;
        }
        this.updateCameraPosition();
    }

    updateCameraPosition() {
        // Convert matplotlib elev/azim to Three.js camera position
        // Matplotlib: z-up, azim from +x counterclockwise
        // Three.js: y-up, need to swap y↔z and negate
        const elevRad = THREE.MathUtils.degToRad(this.elevation);
        const azimRad = THREE.MathUtils.degToRad(this.azimuth);
        const distance = 5;

        // Matplotlib camera in z-up: (cos(e)*cos(a), cos(e)*sin(a), sin(e))
        // Convert to Three.js y-up: swap y↔z, negate new z
        const x = distance * Math.cos(elevRad) * Math.cos(azimRad);
        const y = distance * Math.sin(elevRad);
        const z = -distance * Math.cos(elevRad) * Math.sin(azimRad);

        this.camera.position.set(x, y, z);
        this.camera.lookAt(0, 0, 0);  // Look at origin (shapes are now centered at z=0)
        this.camera.up.set(0, 1, 0);
    }

    reset() {
        this.stopContinuousRotation();
        this.elevation = 20;
        this.azimuth = -35;
        this.updateCameraPosition();
    }

    startRenderLoop() {
        const render = () => {
            this.animationId = requestAnimationFrame(render);
            this.renderer.render(this.scene, this.camera);
        };
        render();
    }

    dispose() {
        this.stopContinuousRotation();

        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }

        if (this.renderer) {
            this.renderer.dispose();
        }

        if (this.wireframeGroup) {
            this.wireframeGroup.traverse(obj => {
                if (obj.geometry) obj.geometry.dispose();
                if (obj.material) obj.material.dispose();
            });
        }

        if (this.wrapper && this.wrapper.parentNode) {
            this.wrapper.parentNode.removeChild(this.wrapper);
        }
    }
}

// Global reference for cleanup
window.currentViewpointReference = null;

// ============================================================
// End of Interactive 3D Dice Reference Classes
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
    const submitBtn = document.getElementById('submit-answer');
    const userAnswerInput = document.getElementById('user-answer');
    const puzzleImage = document.getElementById('puzzle-image');
    const puzzleImageContainer = document.querySelector('.puzzle-image-container');
    const resultMessage = document.getElementById('result-message');
    const totalCount = document.getElementById('total-count');
    const correctCount = document.getElementById('correct-count');
    const accuracyEl = document.getElementById('accuracy');
    const puzzlePrompt = document.getElementById('puzzle-prompt');
    const inputGroup = document.querySelector('.input-group');

    const benchmarkStats = { total: 0, correct: 0 };

    // Session and puzzle type tracking
    let sessionId = 'default';
    let activePuzzleType = null; // Track the currently selected puzzle type
    let typeStats = { total: 0, current: 0, correct: 0, time: 0 }; // Stats for the current type

    let currentPuzzle = null;
    let bingoSelectedCells = [];
    let shadowSelectedCells = [];
    let mirrorSelectedCells = [];
    let squiggleSelectedIndex = null;
    let transformPipelineSelectedIndex = null;
    let selectedGridCells = [];
    let storyboardOrder = [];
    let storyboardSelectedIndices = [];
    let illusionOrder = [];
    let illusionOrderSelectedIndices = [];
    let jigsawPlacements = [];

    // Expose jigsawPlacements globally so agents can programmatically update it
    window.jigsawPlacements = jigsawPlacements;

    let squiggleRevealTimeout = null;
    let colorCipherRevealTimeout = null;
    let redDotTimeout = null;
    let redDotAnswered = false;
    let redDotHits = 0;
    let redDotRequiredHits = 0;
    let redDotTimeoutDuration = 2000;
    let redDotElement = null;
    let spookySizeAnswered = false;
    let spookySizeClickAnswer = null;

    submitBtn.addEventListener('click', submitAnswer);
    userAnswerInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            submitAnswer();
        }
    });

    // Log text input changes
    userAnswerInput.addEventListener('input', (event) => {
        logAction('text_input', { value: event.target.value });
    });

    displayDifficultyStars('Dice_Count');

    // Function to update type stats display
    function updateTypeStatsDisplay() {
        const typeStatsSection = document.getElementById('type-stats-section');
        const typeLabel = document.getElementById('type-label');
        const typePuzzles = document.getElementById('type-puzzles');
        const typeAccuracy = document.getElementById('type-accuracy');
        const typeTime = document.getElementById('type-time');

        if (activePuzzleType) {
            // Show the type stats section
            if (typeStatsSection) {
                typeStatsSection.style.display = 'flex';
            }

            // Update type label
            if (typeLabel) {
                typeLabel.textContent = activePuzzleType.replace(/_/g, ' ');
            }

            // Update type puzzles
            if (typePuzzles) {
                typePuzzles.textContent = `${typeStats.current}/${typeStats.total}`;
            }

            // Update type accuracy
            if (typeAccuracy) {
                const accuracy = typeStats.current > 0
                    ? ((typeStats.correct / typeStats.current) * 100).toFixed(1)
                    : '0.0';
                typeAccuracy.textContent = `${accuracy}%`;
            }

            // Update type time
            if (typeTime) {
                typeTime.textContent = `${typeStats.time.toFixed(1)}s`;
            }
        } else {
            // Hide the type stats section
            if (typeStatsSection) {
                typeStatsSection.style.display = 'none';
            }
        }
    }

    // Fetch and render puzzle types
    fetch('/api/puzzle_types')
        .then(response => response.json())
        .then(data => {
            const selector = document.getElementById('puzzle-type-selector');
            if (selector && data.types) {
                selector.innerHTML = '';
                
                // Add "Random" button
                const randomBtn = document.createElement('button');
                randomBtn.textContent = '🎲 Random';
                randomBtn.className = 'type-btn';
                randomBtn.style.padding = '6px 12px';
                randomBtn.style.border = '1px solid #ddd';
                randomBtn.style.borderRadius = '4px';
                randomBtn.style.cursor = 'pointer';
                randomBtn.style.backgroundColor = '#fff';
                randomBtn.onclick = () => {
                    activePuzzleType = null; // Clear active type for random mode
                    updateTypeStatsDisplay(); // Reset display to global stats
                    loadNewPuzzle();
                };
                selector.appendChild(randomBtn);

                data.types.forEach(type => {
                    const btn = document.createElement('button');
                    btn.textContent = type.replace(/_/g, ' ');
                    btn.className = 'type-btn';
                    btn.style.padding = '6px 12px';
                    btn.style.border = '1px solid #ddd';
                    btn.style.borderRadius = '4px';
                    btn.style.cursor = 'pointer';
                    btn.style.backgroundColor = '#fff';
                    btn.onclick = () => loadNewPuzzle(type);
                    selector.appendChild(btn);
                });
            }
        })
        .catch(err => console.error('Failed to load puzzle types:', err));

    // Check for type, puzzle_index, and seed parameters in URL and load that puzzle initially
    const urlParams = new URLSearchParams(window.location.search);
    const initialType = urlParams.get('type');
    const initialPuzzleIndex = urlParams.get('puzzle_index');
    const initialSeed = urlParams.get('seed');
    loadNewPuzzle(initialType, initialPuzzleIndex, initialSeed);

    function resetInterface() {
        bingoSelectedCells = [];
        shadowSelectedCells = [];
        mirrorSelectedCells = [];
        squiggleSelectedIndex = null;
        transformPipelineSelectedIndex = null;
        storyboardOrder = [];
        storyboardSelectedIndices = [];
        illusionOrder = [];
        illusionOrderSelectedIndices = [];
        jigsawPlacements = [];
        window.jigsawPlacements = jigsawPlacements;

        // Clean up interactive 3D dice reference
        if (window.currentDiceReference) {
            window.currentDiceReference.dispose();
            window.currentDiceReference = null;
        }

        // Clean up interactive 3D viewpoint reference
        if (window.currentViewpointReference) {
            window.currentViewpointReference.dispose();
            window.currentViewpointReference = null;
        }

        if (squiggleRevealTimeout) {
            clearTimeout(squiggleRevealTimeout);
            squiggleRevealTimeout = null;
        }
        if (colorCipherRevealTimeout) {
            clearTimeout(colorCipherRevealTimeout);
            colorCipherRevealTimeout = null;
        }
        if (redDotTimeout) {
            clearTimeout(redDotTimeout);
            redDotTimeout = null;
        }
        redDotAnswered = false;
        redDotHits = 0;
        redDotRequiredHits = 0;
        redDotTimeoutDuration = 2000;
        redDotElement = null;
        spookySizeAnswered = false;

        if (inputGroup) {
            inputGroup.style.display = 'flex';
        }

        userAnswerInput.type = 'text';
        userAnswerInput.value = '';
        userAnswerInput.placeholder = 'Your answer';
        userAnswerInput.style.display = 'block';

        submitBtn.style.display = 'block';
        submitBtn.disabled = false;
        submitBtn.textContent = 'Submit';

        resultMessage.textContent = '';
        resultMessage.className = 'result-message';

        puzzleImageContainer.innerHTML = '';
        puzzleImageContainer.style.display = '';
        puzzleImageContainer.style.width = '';
        puzzleImageContainer.style.maxWidth = '';
        puzzleImageContainer.style.margin = '';
        if (puzzleImageContainer) {
            puzzleImageContainer.classList.remove('adversarial-layout');
        }

        puzzleImage.style.display = 'none';
        puzzleImage.src = '';

        const customSelectors = [
            '.bingo-grid',
            '.bingo-submit',
            '.shadow-plausible-grid',
            '.shadow-submit',
            '.mirror-layout',
            '.mirror-submit',
            '.squiggle-preview',
            '.squiggle-options-grid',
            '.squiggle-submit',
            '.transform-pipeline-container',
            '.transform-pipeline-submit',
            '.color-cipher-preview',
            '.color-cipher-question',
            '.red-dot-area',
            '.trajectory-gif-container',
            '.set-game-rules-container',
            '.storyboard-logic-container',
            '.jigsaw-puzzle-container',
            '.dual-number-input-container'
        ];

        customSelectors.forEach((selector) => {
            document.querySelectorAll(selector).forEach((element) => element.remove());
        });
    }

    function submitRedDotAttempt(redDotAnswer) {
        if (!currentPuzzle || currentPuzzle.input_type !== 'red_dot_click') {
            return;
        }

        const answerData = {
            puzzle_type: currentPuzzle.puzzle_type,
            puzzle_id: currentPuzzle.puzzle_id,
            answer: redDotAnswer
        };
        answerData.elapsed_time = ((Date.now() - (puzzleStartTime || Date.now())) / 1000).toFixed(2);

        fetch('/api/check_answer', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(answerData)
        })
            .then((response) => response.json())
            .then((data) => {
                if (data.status === 'continue') {
                    redDotHits = Number.isFinite(data.hits_completed) ? data.hits_completed : redDotHits;
                    redDotRequiredHits = Number.isFinite(data.required_hits) ? data.required_hits : redDotRequiredHits;
                    redDotTimeoutDuration = Number.isFinite(data.timeout_ms) ? data.timeout_ms : redDotTimeoutDuration;

                    const nextDot = data.next_dot || {};
                    currentPuzzle.dot = nextDot;
                    currentPuzzle.timeout_ms = redDotTimeoutDuration;
                    currentPuzzle.required_hits = redDotRequiredHits;
                    currentPuzzle.hits_completed = redDotHits;
                    if (redDotElement) {
                        if (Number.isFinite(nextDot.diameter)) {
                            redDotElement.style.width = `${nextDot.diameter}px`;
                            redDotElement.style.height = `${nextDot.diameter}px`;
                        }
                        if (Number.isFinite(nextDot.x)) {
                            redDotElement.style.left = `${nextDot.x}px`;
                        }
                        if (Number.isFinite(nextDot.y)) {
                            redDotElement.style.top = `${nextDot.y}px`;
                        }
                        redDotElement.classList.remove('red-dot-hidden');
                    }

                    redDotAnswered = false;
                    displayRedDotProgress();
                    scheduleRedDotTimeout(redDotTimeoutDuration);
                    return;
                }

                benchmarkStats.total += 1;

                if (data.correct) {
                    benchmarkStats.correct += 1;
                    redDotHits = redDotRequiredHits;
                    resultMessage.textContent = 'Correct!';
                    resultMessage.className = 'result-message correct';
                    createFireworks();
                } else {
                    const failureMessage = data.message || 'Incorrect.';
                    resultMessage.textContent = failureMessage;
                    resultMessage.className = 'result-message incorrect';
                    createSadFace();
                }

                updateStats();
                recordBenchmarkResult({
                    puzzle_type: currentPuzzle.puzzle_type,
                    puzzle_id: currentPuzzle.puzzle_id,
                    user_answer: redDotAnswer,
                    correct_answer: data.correct_answer,
                    correct: data.correct,
                    elapsed_time: data.elapsed_time || answerData.elapsed_time,
                    action_sequence: data.action_sequence || actionSequence
                });

                // Stop the timer since puzzle is complete
                stopTimer();

                setTimeout(loadNewPuzzle, 2000);
            })
            .catch((error) => {
                console.error('Error checking red dot answer:', error);
                showError('Error checking answer. Please try again.');
            });
    }

    function renderPuzzleMedia(data) {
        // Handle SVG content (e.g., Map_Parity puzzles)
        if (data.svg_content) {
            const svgWrapper = document.createElement('div');
            svgWrapper.className = 'svg-puzzle-container';
            svgWrapper.innerHTML = data.svg_content;
            puzzleImageContainer.appendChild(svgWrapper);
            return;
        }

        // For Dice_Roll_Path, show interactive 3D dice reference
        if (data.puzzle_type === 'Dice_Roll_Path') {
            // Clean up any existing dice reference
            if (window.currentDiceReference) {
                window.currentDiceReference.dispose();
                window.currentDiceReference = null;
            }

            const diceContainer = document.createElement('div');
            diceContainer.className = 'dice-roll-reference-container';
            puzzleImageContainer.appendChild(diceContainer);

            // Initialize the interactive 3D dice (always starts with 1 on top)
            window.currentDiceReference = new InteractiveDiceReference(diceContainer);
        }

        const mediaPath = data.media_path || data.image_path;
        if (!mediaPath) {
            return;
        }

        const mediaType = (data.media_type || 'image').toLowerCase();
        if (mediaType === 'video') {
            const video = document.createElement('video');
            video.className = 'puzzle-video';
            video.src = mediaPath;
            video.autoplay = true;
            video.loop = true;
            // For Audio_Video_Alignment, audio is essential - don't mute
            video.muted = (data.puzzle_type !== 'Audio_Video_Alignment');
            video.playsInline = true;
            video.controls = true;
            video.setAttribute('preload', 'auto');
            puzzleImageContainer.appendChild(video);

            // For Audio_Video_Alignment, add replay button
            if (data.puzzle_type === 'Audio_Video_Alignment') {
                const replayBtn = document.createElement('button');
                replayBtn.className = 'replay-btn';
                replayBtn.textContent = '🔄 Replay Video';
                replayBtn.style.cssText = 'margin-top: 10px; padding: 8px 16px; cursor: pointer; background: #4CAF50; color: white; border: none; border-radius: 4px;';
                replayBtn.onclick = () => {
                    video.currentTime = 0;
                    video.play();
                };
                puzzleImageContainer.appendChild(replayBtn);
            }
        } else {
            puzzleImage.src = mediaPath;
            puzzleImage.alt = data.media_alt || data.prompt || 'CAPTCHA Puzzle';
            puzzleImage.style.display = 'block';
            puzzleImageContainer.appendChild(puzzleImage);
        }
    }

    function loadNewPuzzle(specificType = null, puzzleIndex = null, seed = null) {
        resetInterface();
        // Clear any stale result from previous puzzle to prevent incorrect detection
        window.lastPuzzleResult = null;
        puzzlePrompt.textContent = 'Loading puzzle...';

        // Determine URL based on whether we're continuing with an active type or selecting a new one
        let url = '/api/get_puzzle?mode=sequential';
        if (specificType) {
            // User selected a specific type
            url = `/api/get_puzzle?type=${specificType}&session_id=${sessionId}`;
            // Add puzzle_index if provided (for deterministic benchmarking)
            if (puzzleIndex !== null) {
                url += `&puzzle_index=${puzzleIndex}`;
            }
            // Add seed if provided (for reproducible puzzle generation)
            if (seed !== null) {
                url += `&seed=${seed}`;
            }
            activePuzzleType = specificType;
        } else if (activePuzzleType) {
            // Continue with the active type
            url = `/api/get_puzzle?continue_active=true&session_id=${sessionId}`;
        }

        fetch(url)
            .then((response) => response.json())
            .then((data) => {
                if (data.error) {
                    throw new Error(data.error);
                }

                currentPuzzle = data;
                // Expose to window for browser automation agents
                window.currentPuzzle = currentPuzzle;
                puzzleStartTime = Date.now();
                actionSequence = [];
                logAction('puzzle_loaded', { puzzle_type: data.puzzle_type, puzzle_id: data.puzzle_id });
                startTimer();

                // Update type stats if available
                if (data.type_stats) {
                    typeStats.total = data.type_stats.total_puzzles || 0;
                    typeStats.current = data.type_stats.current_solved || 0;
                    typeStats.correct = data.type_stats.current_correct || 0;
                    typeStats.time = data.type_stats.current_time || 0;
                    updateTypeStatsDisplay();
                }

                displayDifficultyStars(data.puzzle_type);
                puzzlePrompt.textContent = data.prompt || 'Solve the CAPTCHA puzzle';

                // Show puzzle ID in fixed sequential mode
                const puzzleIdDisplay = document.getElementById('puzzle-id-display');
                const puzzleIdText = document.getElementById('puzzle-id-text');
                if (puzzleIdDisplay && puzzleIdText) {
                    if (data.fixed_sequential_mode && data.puzzle_id) {
                        puzzleIdText.textContent = data.puzzle_id;
                        puzzleIdDisplay.style.display = 'block';
                    } else {
                        puzzleIdDisplay.style.display = 'none';
                    }
                }
                if (puzzleImageContainer) {
                    const isAdversarial = data.puzzle_type === 'Adversarial';
                    puzzleImageContainer.classList.toggle('adversarial-layout', isAdversarial);
                }

                switch (data.input_type) {
                    case 'number':
                        configureNumberPuzzle(data);
                        break;
                    case 'dual_number':
                        configureDualNumberPuzzle(data);
                        break;
                    case 'bingo_swap':
                        setupBingoSwap(data);
                        break;
                    case 'shadow_plausible':
                        setupShadowPlausibleGrid(data);
                break;
            case 'mirror_select':
                setupMirrorSelect(data);
                break;
            case 'squiggle_select':
                setupSquiggleSelect(data);
                break;
            case 'structure_from_motion_select':
                setupGridSelection(data);
                break;
            case 'color_cipher':
                setupColorCipher(data);
                break;
            case 'red_dot_click':
                setupRedDotClick(data);
                break;
            case 'spooky_size_click':
                setupSpookySizeClick(data);
                break;
            case 'storyboard_logic':
                setupStoryboardLogic(data);
                break;
            case 'jigsaw_puzzle':
                setupJigsawPuzzle(data);
                break;
            case 'transform_pipeline_select':
                setupTransformPipelineSelect(data);
                break;
            case 'circle_grid_select':
            case 'circle_grid_direction_select':
            case 'shape_grid_select':
            case 'color_counting_select':
            case 'hole_counting_select':
            case 'rotation_match_select':
            case 'rhythm_select':
            case 'backmost_layer_select':
            case 'shadow_direction_select':
            case 'global_phase_drift_select':
            case 'temporal_continuity_select':
            case 'layered_stack_select':
            case 'illusory_ribbons_select':
            case 'subway_paths_select':
            case 'trajectory_recovery_select':
            case 'set_game_select':
            case 'audio_match_select':
            case 'viewpoint_select':
            case 'box_folding_select':
            case 'illusion_grid_select':
            case 'multi_script_select':
                setupGridSelection(data);
                break;
            case 'illusion_order':
                setupIllusionOrder(data);
                break;
            case 'illusion_count':
                setupIllusionCount(data);
                break;
            case 'map_parity_select':
                setupMapParitySelect(data);
                break;
            default:
                configureTextPuzzle(data);
                break;
        }
            })
            .catch((error) => {
                console.error('Error loading puzzle:', error);
                showError('Unable to load a new puzzle. Please refresh the page.');
            });
    }

    function setupRedDotClick(data) {
        if (inputGroup) {
            inputGroup.style.display = 'none';
        }
        submitBtn.style.display = 'none';

        redDotAnswered = false;
        redDotHits = Number.isFinite(data?.hits_completed) ? data.hits_completed : 0;
        redDotRequiredHits = Number.isFinite(data?.required_hits) ? data.required_hits : 1;

        const areaWidth = Number.isFinite(data?.area?.width) ? data.area.width : 420;
        const areaHeight = Number.isFinite(data?.area?.height) ? data.area.height : 320;
        const dotDiameter = Number.isFinite(data?.dot?.diameter) ? data.dot.diameter : 48;
        const dotX = Number.isFinite(data?.dot?.x) ? data.dot.x : (areaWidth - dotDiameter) / 2;
        const dotY = Number.isFinite(data?.dot?.y) ? data.dot.y : (areaHeight - dotDiameter) / 2;
        const timeoutMs = Number.isFinite(data?.timeout_ms) ? data.timeout_ms : 2000;
        redDotTimeoutDuration = timeoutMs;

        const area = document.createElement('div');
        area.className = 'red-dot-area';
        area.style.width = `${areaWidth}px`;
        area.style.height = `${areaHeight}px`;

        const dot = document.createElement('div');
        dot.className = 'red-dot';
        dot.style.width = `${dotDiameter}px`;
        dot.style.height = `${dotDiameter}px`;
        dot.style.left = `${dotX}px`;
        dot.style.top = `${dotY}px`;

        area.appendChild(dot);
        puzzleImageContainer.appendChild(area);

        redDotElement = dot;

        const handleSuccessClick = (event) => {
            if (redDotAnswered) {
                return;
            }
            event.stopPropagation();
            const areaRect = area.getBoundingClientRect();
            const clickX = event.clientX - areaRect.left;
            const clickY = event.clientY - areaRect.top;

            logAction('red_dot_clicked', { x: clickX, y: clickY, puzzle_type: 'Red_Dot' });

            finalizeRedDotAttempt({
                clicked: true,
                position: {
                    x: Number.isFinite(clickX) ? Number(clickX.toFixed(2)) : clickX,
                    y: Number.isFinite(clickY) ? Number(clickY.toFixed(2)) : clickY
                },
                relative_position: {
                    x: Number((clickX / areaWidth).toFixed(4)),
                    y: Number((clickY / areaHeight).toFixed(4))
                }
            });
        };

        dot.addEventListener('click', handleSuccessClick);

        scheduleRedDotTimeout(timeoutMs);
        displayRedDotProgress();
    }

    function scheduleRedDotTimeout(duration) {
        if (redDotTimeout) {
            clearTimeout(redDotTimeout);
        }
        redDotTimeout = window.setTimeout(() => {
            if (redDotAnswered) {
                return;
            }
            if (redDotElement) {
                redDotElement.classList.add('red-dot-hidden');
            }
            finalizeRedDotAttempt({ clicked: false });
        }, duration);
    }

    function displayRedDotProgress() {
        if (redDotRequiredHits <= 1) {
            resultMessage.textContent = 'Click the red dot before it disappears!';
        } else {
            resultMessage.textContent = `Click the red dot before it disappears! (${redDotHits}/${redDotRequiredHits})`;
        }
        resultMessage.className = 'result-message instruction';
    }

    function setupSpookySizeClick(data) {
        if (inputGroup) {
            inputGroup.style.display = 'none';
        }
        submitBtn.style.display = 'none';

        const canvasWidth = data.canvas_width || 600;
        const canvasHeight = data.canvas_height || 400;

        // Create clickable canvas overlay for the GIF
        const clickArea = document.createElement('div');
        clickArea.className = 'spooky-size-click-area';
        clickArea.style.width = `${canvasWidth}px`;
        clickArea.style.height = `${canvasHeight}px`;
        clickArea.style.position = 'relative';
        clickArea.style.margin = '0 auto';
        clickArea.style.cursor = 'crosshair';
        clickArea.style.border = '2px solid #333';
        clickArea.style.backgroundColor = '#000';

        // Add the GIF as background or img element
        const gifImg = document.createElement('img');
        gifImg.src = data.media_path;
        gifImg.alt = 'Spooky Size Puzzle';
        gifImg.style.width = '100%';
        gifImg.style.height = '100%';
        gifImg.style.display = 'block';
        gifImg.style.pointerEvents = 'none'; // Let clicks pass through to parent

        clickArea.appendChild(gifImg);

        // Handle click
        clickArea.addEventListener('click', (event) => {
            if (spookySizeAnswered) {
                return;
            }

            const rect = clickArea.getBoundingClientRect();
            const clickX = event.clientX - rect.left;
            const clickY = event.clientY - rect.top;

            // Visual feedback
            const marker = document.createElement('div');
            marker.style.position = 'absolute';
            marker.style.left = `${clickX}px`;
            marker.style.top = `${clickY}px`;
            marker.style.width = '20px';
            marker.style.height = '20px';
            marker.style.marginLeft = '-10px';
            marker.style.marginTop = '-10px';
            marker.style.borderRadius = '50%';
            marker.style.border = '3px solid #0078ff';
            marker.style.backgroundColor = 'rgba(0, 120, 255, 0.3)';
            marker.style.pointerEvents = 'none';
            clickArea.appendChild(marker);

            // Disable further clicks
            clickArea.style.pointerEvents = 'none';
            spookySizeAnswered = true;

            // Store answer and submit using shared function
            spookySizeClickAnswer = {
                position: {
                    x: Number(clickX.toFixed(2)),
                    y: Number(clickY.toFixed(2))
                }
            };
            submitAnswer();
        });

        puzzleImageContainer.appendChild(clickArea);
        puzzleImageContainer.style.display = 'block';
    }

    function setupStoryboardLogic(data) {
        if (inputGroup) {
            inputGroup.style.display = 'none';
        }
        submitBtn.style.display = 'none';

        // Initialize order: start with shuffled order to make it interesting
        const images = data.images || [];
        if (!images.length) {
            showError('No storyboard images available.');
            return;
        }

        // Start with images in random order (for challenge)
        storyboardOrder = Array.from({ length: images.length }, (_, i) => i);
        // Shuffle the order
        for (let i = storyboardOrder.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [storyboardOrder[i], storyboardOrder[j]] = [storyboardOrder[j], storyboardOrder[i]];
        }

        // Reset selection
        storyboardSelectedIndices = [];

        const container = document.createElement('div');
        container.className = 'storyboard-logic-container';
        container.style.display = 'flex';
        container.style.flexDirection = 'column';
        container.style.alignItems = 'center';
        container.style.gap = '20px';
        container.style.margin = '20px auto';
        container.style.maxWidth = '900px';

        const instruction = document.createElement('div');
        instruction.style.fontSize = '16px';
        instruction.style.fontWeight = '500';
        instruction.style.marginBottom = '10px';
        instruction.style.textAlign = 'center';
        instruction.textContent = 'Click two images to swap their positions. Arrange them in the correct story sequence.';
        container.appendChild(instruction);

        const imageRow = document.createElement('div');
        imageRow.style.display = 'flex';
        imageRow.style.gap = '15px';
        imageRow.style.justifyContent = 'center';
        imageRow.style.flexWrap = 'nowrap';
        imageRow.style.width = '100%';
        imageRow.style.alignItems = 'flex-start';

        const renderImages = () => {
            imageRow.innerHTML = '';
            storyboardOrder.forEach((imageIndex, position) => {
                const imageWrapper = document.createElement('div');
                imageWrapper.style.position = 'relative';
                imageWrapper.style.display = 'flex';
                imageWrapper.style.flexDirection = 'column';
                imageWrapper.style.alignItems = 'center';
                imageWrapper.style.cursor = 'pointer';
                imageWrapper.style.transition = 'transform 0.2s';
                imageWrapper.dataset.index = imageIndex;
                imageWrapper.dataset.position = position;

                const positionLabel = document.createElement('div');
                positionLabel.style.position = 'absolute';
                positionLabel.style.top = '-25px';
                positionLabel.style.fontSize = '14px';
                positionLabel.style.fontWeight = '600';
                positionLabel.style.color = '#0078ff';
                positionLabel.textContent = `${position + 1}`;
                imageWrapper.appendChild(positionLabel);

                const img = document.createElement('img');
                img.src = images[imageIndex];
                img.alt = `Storyboard image ${imageIndex + 1}`;
                img.style.width = '250px';
                img.style.height = 'auto';
                img.style.maxWidth = '250px';
                img.style.minWidth = '200px';
                img.style.flexShrink = '0';
                img.style.border = 'none';
                img.style.borderRadius = '8px';
                img.draggable = false;
                imageWrapper.appendChild(img);

                // Set initial border styling
                imageWrapper.style.border = '3px solid #333';
                imageWrapper.style.borderRadius = '8px';
                imageWrapper.style.boxShadow = '0 2px 8px rgba(0,0,0,0.1)';
                imageWrapper.style.padding = '0';
                
                // Check if this position is selected
                const isSelected = storyboardSelectedIndices.includes(position);
                if (isSelected) {
                    imageWrapper.style.border = '3px solid #0078ff';
                    imageWrapper.style.boxShadow = '0 0 0 3px rgba(0, 120, 255, 0.3), 0 2px 8px rgba(0,0,0,0.1)';
                }

                imageWrapper.addEventListener('click', () => {
                    const clickedPosition = position;
                    
                    // If already selected, deselect it
                    if (storyboardSelectedIndices.includes(clickedPosition)) {
                        storyboardSelectedIndices = storyboardSelectedIndices.filter(idx => idx !== clickedPosition);
                        renderImages();
                        return;
                    }
                    
                    // If no selection yet, select this position
                    if (storyboardSelectedIndices.length === 0) {
                        storyboardSelectedIndices.push(clickedPosition);
                        renderImages();
                        return;
                    }
                    
                    // If one position is already selected, swap with this one
                    if (storyboardSelectedIndices.length === 1) {
                        const firstPos = storyboardSelectedIndices[0];
                        const secondPos = clickedPosition;
                        
                        // Swap the images at these positions
                        const temp = storyboardOrder[firstPos];
                        storyboardOrder[firstPos] = storyboardOrder[secondPos];
                        storyboardOrder[secondPos] = temp;
                        
                        // Clear selection
                        storyboardSelectedIndices = [];
                        renderImages();
                    }
                });

                imageWrapper.addEventListener('mouseenter', () => {
                    if (!storyboardSelectedIndices.includes(position)) {
                        imageWrapper.style.transform = 'scale(1.05)';
                    }
                });

                imageWrapper.addEventListener('mouseleave', () => {
                    imageWrapper.style.transform = 'scale(1)';
                });

                imageRow.appendChild(imageWrapper);
            });
        };

        renderImages();
        container.appendChild(imageRow);

        const submitSection = document.createElement('div');
        submitSection.style.marginTop = '20px';

        const storyboardSubmitBtn = document.createElement('button');
        storyboardSubmitBtn.textContent = 'Submit Order';
        storyboardSubmitBtn.className = 'submit-storyboard';
        storyboardSubmitBtn.style.padding = '12px 24px';
        storyboardSubmitBtn.style.fontSize = '16px';
        storyboardSubmitBtn.style.fontWeight = '600';
        storyboardSubmitBtn.style.backgroundColor = '#0078ff';
        storyboardSubmitBtn.style.color = 'white';
        storyboardSubmitBtn.style.border = 'none';
        storyboardSubmitBtn.style.borderRadius = '6px';
        storyboardSubmitBtn.style.cursor = 'pointer';
        storyboardSubmitBtn.style.transition = 'background-color 0.2s';
        storyboardSubmitBtn.type = 'button';

        storyboardSubmitBtn.addEventListener('mouseenter', () => {
            storyboardSubmitBtn.style.backgroundColor = '#0056b3';
        });

        storyboardSubmitBtn.addEventListener('mouseleave', () => {
            storyboardSubmitBtn.style.backgroundColor = '#0078ff';
        });

        storyboardSubmitBtn.addEventListener('click', () => {
            storyboardSubmitBtn.disabled = true;
            storyboardSubmitBtn.textContent = 'Processing...';
            submitAnswer();
        });

        submitSection.appendChild(storyboardSubmitBtn);
        container.appendChild(submitSection);

        puzzleImageContainer.appendChild(container);
        puzzleImageContainer.style.display = 'block';
    }

    function setupIllusionOrder(data) {
        if (inputGroup) {
            inputGroup.style.display = 'none';
        }
        submitBtn.style.display = 'none';

        // Get images and order count from data
        const images = data.images || [];
        const orderCount = data.order_count || 3;
        if (!images.length) {
            showError('No illusion images available.');
            return;
        }

        // Track which images are placed in ranking slots
        // rankingSlots[0] = index of image in "Smallest" slot, etc.
        let rankingSlots = [null, null, null];
        let selectedImageIndex = null;

        const container = document.createElement('div');
        container.className = 'illusion-order-container';
        container.style.display = 'flex';
        container.style.flexDirection = 'column';
        container.style.alignItems = 'center';
        container.style.gap = '20px';
        container.style.margin = '20px auto';
        container.style.maxWidth = '900px';

        const instruction = document.createElement('div');
        instruction.style.fontSize = '16px';
        instruction.style.fontWeight = '500';
        instruction.style.marginBottom = '10px';
        instruction.style.textAlign = 'center';
        instruction.innerHTML = `Click an image above, then click a ranking slot below to place it.<br>Find the 3 animals and order them by size.`;
        container.appendChild(instruction);

        // Create 3x3 grid layout for all images
        const imageGrid = document.createElement('div');
        imageGrid.style.display = 'grid';
        imageGrid.style.gridTemplateColumns = 'repeat(3, 1fr)';
        imageGrid.style.gap = '12px';
        imageGrid.style.justifyContent = 'center';
        imageGrid.style.maxWidth = '550px';

        // Create ranking slots section
        const rankingSection = document.createElement('div');
        rankingSection.style.marginTop = '25px';
        rankingSection.style.padding = '20px';
        rankingSection.style.backgroundColor = '#f0f7ff';
        rankingSection.style.borderRadius = '12px';
        rankingSection.style.border = '2px dashed #0078ff';

        const rankingTitle = document.createElement('div');
        rankingTitle.style.fontSize = '15px';
        rankingTitle.style.fontWeight = '600';
        rankingTitle.style.marginBottom = '15px';
        rankingTitle.style.textAlign = 'center';
        rankingTitle.style.color = '#0078ff';
        rankingTitle.textContent = '↓ Place animals here in order (Smallest → Largest) ↓';
        rankingSection.appendChild(rankingTitle);

        const rankingSlotsContainer = document.createElement('div');
        rankingSlotsContainer.style.display = 'flex';
        rankingSlotsContainer.style.justifyContent = 'center';
        rankingSlotsContainer.style.gap = '20px';

        const slotLabels = ['1st (Smallest)', '2nd (Medium)', '3rd (Largest)'];

        const renderAll = () => {
            // Render image grid
            imageGrid.innerHTML = '';
            images.forEach((imageSrc, index) => {
                const imageWrapper = document.createElement('div');
                imageWrapper.style.position = 'relative';
                imageWrapper.style.display = 'flex';
                imageWrapper.style.flexDirection = 'column';
                imageWrapper.style.alignItems = 'center';
                imageWrapper.style.cursor = 'pointer';
                imageWrapper.style.transition = 'all 0.2s';
                imageWrapper.dataset.index = index;

                const img = document.createElement('img');
                img.src = imageSrc;
                img.alt = `Image ${index + 1}`;
                img.style.width = '160px';
                img.style.height = '160px';
                img.style.objectFit = 'cover';
                img.style.borderRadius = '8px';
                img.draggable = false;

                // Check if this image is already placed in a slot
                const placedInSlot = rankingSlots.indexOf(index);
                if (placedInSlot !== -1) {
                    imageWrapper.style.opacity = '0.3';
                    imageWrapper.style.border = '3px solid #ccc';
                    img.style.filter = 'grayscale(50%)';
                } else if (selectedImageIndex === index) {
                    imageWrapper.style.border = '3px solid #ff6600';
                    imageWrapper.style.boxShadow = '0 0 0 3px rgba(255, 102, 0, 0.3)';
                } else {
                    imageWrapper.style.border = '3px solid #333';
                }
                imageWrapper.style.borderRadius = '8px';
                imageWrapper.style.boxShadow = placedInSlot === -1 ? '0 2px 8px rgba(0,0,0,0.1)' : 'none';

                imageWrapper.appendChild(img);

                // Click to select (only if not already placed)
                if (placedInSlot === -1) {
                    imageWrapper.addEventListener('click', () => {
                        if (selectedImageIndex === index) {
                            selectedImageIndex = null;
                        } else {
                            selectedImageIndex = index;
                        }
                        renderAll();
                    });

                    imageWrapper.addEventListener('mouseenter', () => {
                        if (selectedImageIndex !== index) {
                            imageWrapper.style.transform = 'scale(1.05)';
                        }
                    });

                    imageWrapper.addEventListener('mouseleave', () => {
                        imageWrapper.style.transform = 'scale(1)';
                    });
                }

                imageGrid.appendChild(imageWrapper);
            });

            // Render ranking slots
            rankingSlotsContainer.innerHTML = '';
            slotLabels.forEach((label, slotIndex) => {
                const slotWrapper = document.createElement('div');
                slotWrapper.style.display = 'flex';
                slotWrapper.style.flexDirection = 'column';
                slotWrapper.style.alignItems = 'center';
                slotWrapper.style.gap = '8px';

                const slotLabel = document.createElement('div');
                slotLabel.style.fontSize = '13px';
                slotLabel.style.fontWeight = '600';
                slotLabel.style.color = '#555';
                slotLabel.textContent = label;
                slotWrapper.appendChild(slotLabel);

                const slot = document.createElement('div');
                slot.style.width = '140px';
                slot.style.height = '140px';
                slot.style.border = '3px dashed #0078ff';
                slot.style.borderRadius = '8px';
                slot.style.display = 'flex';
                slot.style.alignItems = 'center';
                slot.style.justifyContent = 'center';
                slot.style.backgroundColor = '#fff';
                slot.style.cursor = 'pointer';
                slot.style.transition = 'all 0.2s';

                const placedIndex = rankingSlots[slotIndex];
                if (placedIndex !== null) {
                    // Show placed image
                    const placedImg = document.createElement('img');
                    placedImg.src = images[placedIndex];
                    placedImg.style.width = '130px';
                    placedImg.style.height = '130px';
                    placedImg.style.objectFit = 'cover';
                    placedImg.style.borderRadius = '6px';
                    slot.appendChild(placedImg);
                    slot.style.border = '3px solid #28a745';
                    slot.style.backgroundColor = '#f0fff0';

                    // Click to remove
                    slot.addEventListener('click', () => {
                        rankingSlots[slotIndex] = null;
                        renderAll();
                    });
                } else {
                    // Empty slot
                    const placeholder = document.createElement('div');
                    placeholder.style.color = '#aaa';
                    placeholder.style.fontSize = '30px';
                    placeholder.textContent = '?';
                    slot.appendChild(placeholder);

                    // Click to place selected image
                    slot.addEventListener('click', () => {
                        if (selectedImageIndex !== null) {
                            rankingSlots[slotIndex] = selectedImageIndex;
                            selectedImageIndex = null;
                            renderAll();
                        }
                    });

                    slot.addEventListener('mouseenter', () => {
                        if (selectedImageIndex !== null) {
                            slot.style.backgroundColor = '#e6f3ff';
                            slot.style.borderColor = '#ff6600';
                        }
                    });

                    slot.addEventListener('mouseleave', () => {
                        slot.style.backgroundColor = '#fff';
                        slot.style.borderColor = '#0078ff';
                    });
                }

                slotWrapper.appendChild(slot);
                rankingSlotsContainer.appendChild(slotWrapper);
            });
        };

        renderAll();
        container.appendChild(imageGrid);
        rankingSection.appendChild(rankingSlotsContainer);
        container.appendChild(rankingSection);

        const submitSection = document.createElement('div');
        submitSection.style.marginTop = '20px';

        const orderSubmitBtn = document.createElement('button');
        orderSubmitBtn.textContent = 'Submit Order';
        orderSubmitBtn.className = 'submit-illusion-order';
        orderSubmitBtn.style.padding = '12px 24px';
        orderSubmitBtn.style.fontSize = '16px';
        orderSubmitBtn.style.fontWeight = '600';
        orderSubmitBtn.style.backgroundColor = '#0078ff';
        orderSubmitBtn.style.color = 'white';
        orderSubmitBtn.style.border = 'none';
        orderSubmitBtn.style.borderRadius = '6px';
        orderSubmitBtn.style.cursor = 'pointer';
        orderSubmitBtn.style.transition = 'background-color 0.2s';
        orderSubmitBtn.type = 'button';

        orderSubmitBtn.addEventListener('mouseenter', () => {
            orderSubmitBtn.style.backgroundColor = '#0056b3';
        });

        orderSubmitBtn.addEventListener('mouseleave', () => {
            orderSubmitBtn.style.backgroundColor = '#0078ff';
        });

        orderSubmitBtn.addEventListener('click', () => {
            // Check if all slots are filled
            if (rankingSlots.includes(null)) {
                alert('Please place an image in each ranking slot before submitting.');
                return;
            }
            // Store the answer as the indices in order
            illusionOrder = rankingSlots.slice();
            orderSubmitBtn.disabled = true;
            orderSubmitBtn.textContent = 'Processing...';
            submitAnswer();
        });

        submitSection.appendChild(orderSubmitBtn);
        container.appendChild(submitSection);

        puzzleImageContainer.appendChild(container);
        puzzleImageContainer.style.display = 'block';
    }

    function setupIllusionCount(data) {
        if (inputGroup) {
            inputGroup.style.display = 'block';
        }
        submitBtn.style.display = 'inline-block';

        // Get cell images from data
        const cellImages = data.option_images || [];
        const gridSize = data.grid_size || [4, 4];
        if (!cellImages.length) {
            showError('No illusion images available for counting.');
            return;
        }

        const container = document.createElement('div');
        container.className = 'illusion-count-container';
        container.style.display = 'flex';
        container.style.flexDirection = 'column';
        container.style.alignItems = 'center';
        container.style.gap = '20px';
        container.style.margin = '20px auto';
        container.style.maxWidth = '800px';

        const instruction = document.createElement('div');
        instruction.style.fontSize = '16px';
        instruction.style.fontWeight = '500';
        instruction.style.marginBottom = '10px';
        instruction.style.textAlign = 'center';
        instruction.textContent = 'Look at all images and count as instructed. Enter your answer below.';
        container.appendChild(instruction);

        // Create grid layout
        const imageGrid = document.createElement('div');
        imageGrid.style.display = 'grid';
        imageGrid.style.gridTemplateColumns = `repeat(${gridSize[1]}, 1fr)`;
        imageGrid.style.gap = '10px';
        imageGrid.style.justifyContent = 'center';
        imageGrid.style.maxWidth = '700px';

        cellImages.forEach((imageSrc, index) => {
            const imageWrapper = document.createElement('div');
            imageWrapper.style.position = 'relative';
            imageWrapper.style.display = 'flex';
            imageWrapper.style.flexDirection = 'column';
            imageWrapper.style.alignItems = 'center';

            const img = document.createElement('img');
            img.src = imageSrc;
            img.alt = `Cell ${index + 1}`;
            img.style.width = '150px';
            img.style.height = '150px';
            img.style.objectFit = 'cover';
            img.style.border = '2px solid #333';
            img.style.borderRadius = '6px';
            img.draggable = false;
            imageWrapper.appendChild(img);

            imageGrid.appendChild(imageWrapper);
        });

        container.appendChild(imageGrid);
        puzzleImageContainer.appendChild(container);
        puzzleImageContainer.style.display = 'block';
    }

    function setupMapParitySelect(data) {
        if (inputGroup) {
            inputGroup.style.display = 'none';
        }
        submitBtn.style.display = 'none';

        // Create container for SVG and buttons
        const container = document.createElement('div');
        container.className = 'map-parity-container';
        container.style.display = 'flex';
        container.style.flexDirection = 'column';
        container.style.alignItems = 'center';
        container.style.gap = '20px';
        container.style.margin = '20px auto';
        container.style.maxWidth = '500px';

        // Add SVG content
        if (data.svg_content) {
            const svgWrapper = document.createElement('div');
            svgWrapper.className = 'map-parity-svg';
            svgWrapper.innerHTML = data.svg_content;
            svgWrapper.style.border = '2px solid #333';
            svgWrapper.style.borderRadius = '8px';
            svgWrapper.style.padding = '10px';
            svgWrapper.style.backgroundColor = '#fff';
            container.appendChild(svgWrapper);
        }

        // Create button container
        const buttonContainer = document.createElement('div');
        buttonContainer.className = 'map-parity-buttons';
        buttonContainer.style.display = 'flex';
        buttonContainer.style.gap = '20px';
        buttonContainer.style.justifyContent = 'center';
        buttonContainer.style.marginTop = '15px';

        const options = data.options || ['odd', 'even'];
        options.forEach((option) => {
            const btn = document.createElement('button');
            btn.className = 'map-parity-btn';
            btn.textContent = option.toUpperCase();
            btn.style.padding = '15px 40px';
            btn.style.fontSize = '18px';
            btn.style.fontWeight = 'bold';
            btn.style.cursor = 'pointer';
            btn.style.border = '2px solid #333';
            btn.style.borderRadius = '8px';
            btn.style.backgroundColor = option === 'odd' ? '#FF9800' : '#2196F3';
            btn.style.color = '#fff';
            btn.style.minWidth = '120px';
            btn.style.transition = 'transform 0.1s, box-shadow 0.1s';

            btn.addEventListener('mouseenter', () => {
                btn.style.transform = 'scale(1.05)';
                btn.style.boxShadow = '0 4px 12px rgba(0,0,0,0.3)';
            });
            btn.addEventListener('mouseleave', () => {
                btn.style.transform = 'scale(1)';
                btn.style.boxShadow = 'none';
            });

            btn.addEventListener('click', () => {
                logAction('map_parity_selected', { answer: option, puzzle_type: 'Map_Parity' });
                submitMapParityAnswer(option);
            });

            buttonContainer.appendChild(btn);
        });

        container.appendChild(buttonContainer);
        puzzleImageContainer.appendChild(container);
        puzzleImageContainer.style.display = 'block';
    }

    function submitMapParityAnswer(answer) {
        const elapsedTime = (Date.now() - puzzleStartTime) / 1000;

        fetch('/api/check_answer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                puzzle_id: currentPuzzle.puzzle_id,
                puzzle_type: currentPuzzle.puzzle_type,
                answer: answer,
                elapsed_time: elapsedTime,
                action_sequence: actionSequence
            })
        })
            .then((response) => response.json())
            .then((data) => {
                if (data.error) {
                    showError(data.error);
                    return;
                }

                // Expose result to window for browser automation agents
                window.lastPuzzleResult = {
                    puzzle_type: currentPuzzle.puzzle_type,
                    puzzle_id: currentPuzzle.puzzle_id,
                    correct: data.correct,
                    correct_answer: data.correct_answer,
                    timestamp: Date.now()
                };

                if (data.correct) {
                    benchmarkStats.correct++;
                    typeStats.correct++;
                    resultMessage.textContent = '✓ Correct!';
                    resultMessage.className = 'result-message correct';
                    createFireworks();
                } else {
                    const correctAnswer = data.correct_answer || 'unknown';
                    resultMessage.textContent = `✗ Incorrect. The answer was: ${correctAnswer}`;
                    resultMessage.className = 'result-message incorrect';
                    createSadFace();
                }

                benchmarkStats.total++;
                typeStats.current++;
                typeStats.time += elapsedTime;
                updateStats();
                updateTypeStatsDisplay();

                recordBenchmarkResult({
                    puzzle_type: currentPuzzle.puzzle_type,
                    puzzle_id: currentPuzzle.puzzle_id,
                    user_answer: answer,
                    correct_answer: data.correct_answer,
                    correct: data.correct,
                    elapsed_time: elapsedTime,
                    action_sequence: actionSequence
                });

                stopTimer();
                setTimeout(loadNewPuzzle, 2000);
            })
            .catch((error) => {
                console.error('Error checking map parity answer:', error);
                showError('Error checking answer. Please try again.');
            });
    }

    function setupJigsawPuzzle(data) {
        if (inputGroup) {
            inputGroup.style.display = 'none';
        }
        submitBtn.style.display = 'none';

        const pieces = data.pieces || [];
        const gridSize = data.grid_size || [2, 2];
        const pieceSize = data.piece_size || 150;
        const correctPositions = data.correct_positions || [];
        const referenceImage = data.reference_image;
        // shuffled_order maps display index -> original piece_index
        // If not provided, assume pieces are in original order (0, 1, 2, ...)
        const shuffledOrder = data.shuffled_order || pieces.map((_, i) => i);

        if (!pieces.length) {
            showError('No puzzle pieces available.');
            return;
        }

        // Initialize placements - all pieces start unplaced
        jigsawPlacements = [];
        window.jigsawPlacements = jigsawPlacements;

        const container = document.createElement('div');
        container.className = 'jigsaw-puzzle-container';
        container.style.display = 'flex';
        container.style.flexDirection = 'column';
        container.style.alignItems = 'center';
        container.style.gap = '20px';
        container.style.margin = '20px auto';
        container.style.maxWidth = '900px';

        // Reference image (optional hint)
        if (referenceImage) {
            const referenceSection = document.createElement('div');
            referenceSection.style.textAlign = 'center';
            referenceSection.style.marginBottom = '10px';
            
            const referenceLabel = document.createElement('div');
            referenceLabel.style.fontSize = '14px';
            referenceLabel.style.fontWeight = '500';
            referenceLabel.style.marginBottom = '5px';
            referenceLabel.textContent = 'Reference image:';
            referenceSection.appendChild(referenceLabel);

            const refImg = document.createElement('img');
            refImg.src = referenceImage;
            refImg.alt = 'Jigsaw puzzle reference';
            refImg.style.maxWidth = `${pieceSize * gridSize[1]}px`;
            refImg.style.height = 'auto';
            refImg.style.border = '2px solid #333';
            refImg.style.borderRadius = '8px';
            refImg.style.opacity = '0.7';
            refImg.draggable = false;
            referenceSection.appendChild(refImg);
            container.appendChild(referenceSection);
        }

        // Puzzle grid area
        const gridContainer = document.createElement('div');
        gridContainer.style.display = 'grid';
        gridContainer.style.gridTemplateColumns = `repeat(${gridSize[1]}, ${pieceSize}px)`;
        gridContainer.style.gridTemplateRows = `repeat(${gridSize[0]}, ${pieceSize}px)`;
        gridContainer.style.gap = '2px';
        gridContainer.style.border = '3px solid #333';
        gridContainer.style.padding = '5px';
        gridContainer.style.backgroundColor = '#f0f0f0';
        gridContainer.style.borderRadius = '8px';
        gridContainer.id = 'jigsaw-grid';

        // Create grid cells
        const gridCells = [];
        for (let row = 0; row < gridSize[0]; row++) {
            for (let col = 0; col < gridSize[1]; col++) {
                const cell = document.createElement('div');
                cell.className = 'jigsaw-grid-cell';
                cell.id = `jigsaw-cell-${row}-${col}`;
                cell.dataset.row = row;
                cell.dataset.col = col;
                cell.style.width = `${pieceSize}px`;
                cell.style.height = `${pieceSize}px`;
                cell.style.border = '2px dashed #ccc';
                cell.style.borderRadius = '4px';
                cell.style.backgroundColor = '#fff';
                cell.style.display = 'flex';
                cell.style.alignItems = 'center';
                cell.style.justifyContent = 'center';
                cell.style.position = 'relative';
                cell.style.transition = 'background-color 0.2s';

                // Drop zone
                cell.addEventListener('dragover', (e) => {
                    e.preventDefault();
                    if (!cell.querySelector('.jigsaw-piece')) {
                        cell.style.backgroundColor = '#e8f4f8';
                    }
                });

                cell.addEventListener('dragleave', () => {
                    if (!cell.querySelector('.jigsaw-piece')) {
                        cell.style.backgroundColor = '#fff';
                    }
                });

                cell.addEventListener('drop', (e) => {
                    e.preventDefault();
                    cell.style.backgroundColor = '#fff';

                    // displayIndex is the index in the shuffled pieces array
                    const displayIndex = parseInt(e.dataTransfer.getData('text/plain'));
                    // originalPieceIndex is the actual piece_index that matches correct_positions
                    const originalPieceIndex = shuffledOrder[displayIndex];
                    const row = parseInt(cell.dataset.row);
                    const col = parseInt(cell.dataset.col);

                    logAction('jigsaw_piece_dropped', { piece_index: originalPieceIndex, display_index: displayIndex, to_row: row, to_col: col });

                    // If cell already has a piece, don't replace it
                    if (cell.querySelector('.jigsaw-piece')) {
                        return;
                    }

                    // Find existing placement for this piece (by original piece index)
                    const existingPlacementIdx = jigsawPlacements.findIndex(p => p.piece_index === originalPieceIndex);

                    // If piece was already placed in a different cell, clear that cell
                    if (existingPlacementIdx !== -1) {
                        const oldPlacement = jigsawPlacements[existingPlacementIdx];
                        const oldRow = parseInt(oldPlacement.grid_row);
                        const oldCol = parseInt(oldPlacement.grid_col);
                        // Only clear if it's a different cell
                        if (oldRow !== row || oldCol !== col) {
                            const oldCell = document.querySelector(`.jigsaw-grid-cell[data-row="${oldRow}"][data-col="${oldCol}"]`);
                            if (oldCell && oldCell !== cell) {
                                oldCell.innerHTML = '';
                            }
                            // Update the placement to new position
                            jigsawPlacements[existingPlacementIdx] = {
                                piece_index: originalPieceIndex,
                                grid_row: row,
                                grid_col: col
                            };
                        } else {
                            // Same cell, no change needed
                            return;
                        }
                    } else {
                        // New placement - add to array
                        jigsawPlacements.push({
                            piece_index: originalPieceIndex,
                            grid_row: row,
                            grid_col: col
                        });
                    }

                    // Remove piece from tray if it was there (use displayIndex for DOM lookup)
                    const trayPiece = document.querySelector(`.jigsaw-tray-piece[data-piece-index="${displayIndex}"]`);
                    if (trayPiece) {
                        trayPiece.remove();
                    }

                    // Place piece in this cell (use displayIndex to get image source)
                    const pieceImg = document.createElement('img');
                    pieceImg.src = pieces[displayIndex];
                    pieceImg.className = 'jigsaw-piece';
                    pieceImg.style.width = '100%';
                    pieceImg.style.height = '100%';
                    pieceImg.style.objectFit = 'contain';
                    pieceImg.draggable = true;
                    pieceImg.dataset.pieceIndex = displayIndex;
                    pieceImg.dataset.originalPieceIndex = originalPieceIndex;

                    // Clear cell and add piece
                    cell.innerHTML = '';
                    cell.appendChild(pieceImg);

                    // Make piece draggable again
                    pieceImg.addEventListener('dragstart', (e) => {
                        e.dataTransfer.setData('text/plain', displayIndex.toString());
                        e.dataTransfer.effectAllowed = 'move';
                        logAction('jigsaw_drag_start', { piece_index: originalPieceIndex, from_cell: cellIndex });
                    });

                    // Allow removing piece by dragging to tray
                    pieceImg.addEventListener('dragend', (e) => {
                        // Check if dropped outside grid
                        setTimeout(() => {
                            const dropTarget = document.elementFromPoint(e.clientX, e.clientY);
                            if (!dropTarget?.closest('.jigsaw-grid-cell')) {
                                // Return to tray - remove from cell
                                cell.innerHTML = '';
                                const placementIdx = jigsawPlacements.findIndex(p => p.piece_index === originalPieceIndex);
                                if (placementIdx !== -1) {
                                    jigsawPlacements.splice(placementIdx, 1);
                                }
                                renderPieces();
                            }
                        }, 100);
                    });
                });

                gridContainer.appendChild(cell);
                gridCells.push(cell);
            }
        }

        container.appendChild(gridContainer);

        // Pieces tray
        const trayContainer = document.createElement('div');
        trayContainer.className = 'jigsaw-tray';
        trayContainer.style.display = 'flex';
        trayContainer.style.flexWrap = 'wrap';
        trayContainer.style.gap = '10px';
        trayContainer.style.justifyContent = 'center';
        trayContainer.style.marginTop = '20px';
        trayContainer.style.padding = '15px';
        trayContainer.style.border = '2px dashed #ccc';
        trayContainer.style.borderRadius = '8px';
        trayContainer.style.backgroundColor = '#fafafa';
        trayContainer.style.minHeight = '100px';

        const trayLabel = document.createElement('div');
        trayLabel.style.width = '100%';
        trayLabel.style.textAlign = 'center';
        trayLabel.style.fontSize = '14px';
        trayLabel.style.fontWeight = '500';
        trayLabel.style.marginBottom = '10px';
        trayLabel.textContent = 'Drag pieces from here to the grid above';
        trayContainer.appendChild(trayLabel);

        const renderPieces = () => {
            // Clear tray
            const existingPieces = trayContainer.querySelectorAll('.jigsaw-tray-piece');
            existingPieces.forEach(p => p.remove());

            // Show pieces that are not placed (jigsawPlacements uses original piece indices)
            const placedOriginalIndices = new Set(jigsawPlacements.map(p => p.piece_index));

            // displayIndex is the index in shuffled pieces array
            pieces.forEach((pieceSrc, displayIndex) => {
                // Get original piece index for this display position
                const originalPieceIndex = shuffledOrder[displayIndex];
                // Check if this piece (by original index) is already placed
                if (!placedOriginalIndices.has(originalPieceIndex)) {
                    const pieceWrapper = document.createElement('div');
                    pieceWrapper.className = 'jigsaw-tray-piece';
                    pieceWrapper.id = `jigsaw-piece-${displayIndex}`;
                    pieceWrapper.dataset.pieceIndex = displayIndex;
                    pieceWrapper.dataset.originalPieceIndex = originalPieceIndex;
                    pieceWrapper.style.width = `${pieceSize * 0.6}px`;
                    pieceWrapper.style.height = `${pieceSize * 0.6}px`;
                    pieceWrapper.style.cursor = 'grab';
                    pieceWrapper.style.border = '2px solid #333';
                    pieceWrapper.style.borderRadius = '4px';
                    pieceWrapper.style.overflow = 'hidden';
                    pieceWrapper.style.transition = 'transform 0.2s';
                    pieceWrapper.style.backgroundColor = '#fff';

                    const pieceImg = document.createElement('img');
                    pieceImg.src = pieceSrc;
                    pieceImg.style.width = '100%';
                    pieceImg.style.height = '100%';
                    pieceImg.style.objectFit = 'contain';
                    pieceImg.draggable = true;
                    pieceImg.dataset.pieceIndex = displayIndex;

                    pieceWrapper.appendChild(pieceImg);
                    trayContainer.appendChild(pieceWrapper);

                    pieceImg.addEventListener('dragstart', (e) => {
                        // Pass displayIndex - drop handler will convert to original index
                        e.dataTransfer.setData('text/plain', displayIndex.toString());
                        e.dataTransfer.effectAllowed = 'move';
                        pieceWrapper.style.opacity = '0.5';
                    });

                    pieceImg.addEventListener('dragend', () => {
                        pieceWrapper.style.opacity = '1';
                    });

                    pieceWrapper.addEventListener('mouseenter', () => {
                        pieceWrapper.style.transform = 'scale(1.1)';
                    });

                    pieceWrapper.addEventListener('mouseleave', () => {
                        pieceWrapper.style.transform = 'scale(1)';
                    });
                }
            });
        };

        renderPieces();
        container.appendChild(trayContainer);

        // Submit button
        const submitSection = document.createElement('div');
        submitSection.style.marginTop = '20px';

        const jigsawSubmitBtn = document.createElement('button');
        jigsawSubmitBtn.id = 'jigsaw-submit';
        jigsawSubmitBtn.textContent = 'Submit Puzzle';
        jigsawSubmitBtn.className = 'submit-jigsaw';
        jigsawSubmitBtn.style.padding = '12px 24px';
        jigsawSubmitBtn.style.fontSize = '16px';
        jigsawSubmitBtn.style.fontWeight = '600';
        jigsawSubmitBtn.style.backgroundColor = '#0078ff';
        jigsawSubmitBtn.style.color = 'white';
        jigsawSubmitBtn.style.border = 'none';
        jigsawSubmitBtn.style.borderRadius = '6px';
        jigsawSubmitBtn.style.cursor = 'pointer';
        jigsawSubmitBtn.style.transition = 'background-color 0.2s';
        jigsawSubmitBtn.type = 'button';

        jigsawSubmitBtn.addEventListener('mouseenter', () => {
            jigsawSubmitBtn.style.backgroundColor = '#0056b3';
        });

        jigsawSubmitBtn.addEventListener('mouseleave', () => {
            jigsawSubmitBtn.style.backgroundColor = '#0078ff';
        });

        jigsawSubmitBtn.addEventListener('click', () => {
            // Allow empty submissions - backend will mark as incorrect and move to next puzzle
            jigsawSubmitBtn.disabled = true;
            jigsawSubmitBtn.textContent = 'Processing...';
            submitAnswer();
        });

        submitSection.appendChild(jigsawSubmitBtn);
        container.appendChild(submitSection);

        puzzleImageContainer.appendChild(container);
        puzzleImageContainer.style.display = 'block';
    }

    function configureNumberPuzzle(data) {
        if (inputGroup) {
            inputGroup.style.display = 'flex';
        }

        userAnswerInput.type = 'number';
        userAnswerInput.value = '';
        userAnswerInput.placeholder = 'Enter total';

        submitBtn.style.display = 'block';
        submitBtn.disabled = false;
        submitBtn.textContent = 'Submit';

        renderPuzzleMedia(data);
    }

    function configureDualNumberPuzzle(data) {
        // Show the input group (contains both input and submit button)
        if (inputGroup) {
            inputGroup.style.display = 'flex';
        }

        // Hide the default single input and clear it
        userAnswerInput.style.display = 'none';
        userAnswerInput.value = '';

        // Remove any existing dual input container
        const existingContainer = document.querySelector('.dual-number-input-container');
        if (existingContainer) {
            existingContainer.remove();
        }

        // Create custom container for dual number inputs
        const dualInputContainer = document.createElement('div');
        dualInputContainer.className = 'dual-number-input-container';
        dualInputContainer.style.display = 'flex';
        dualInputContainer.style.gap = '15px';
        dualInputContainer.style.alignItems = 'center';
        dualInputContainer.style.justifyContent = 'center';

        // Create first number input
        const input1 = document.createElement('input');
        input1.type = 'number';
        input1.id = 'dual-number-input-1';
        input1.placeholder = 'Count 1';
        input1.className = 'answer-input';
        input1.style.width = '120px';
        input1.min = '0';

        // Create comma separator
        const separator = document.createElement('span');
        separator.textContent = ',';
        separator.style.fontSize = '24px';
        separator.style.fontWeight = 'bold';
        separator.style.color = '#333';

        // Create second number input
        const input2 = document.createElement('input');
        input2.type = 'number';
        input2.id = 'dual-number-input-2';
        input2.placeholder = 'Count 2';
        input2.className = 'answer-input';
        input2.style.width = '120px';
        input2.min = '0';

        // Assemble container
        dualInputContainer.appendChild(input1);
        dualInputContainer.appendChild(separator);
        dualInputContainer.appendChild(input2);

        // Insert the dual input container before the submit button in the input-group
        inputGroup.insertBefore(dualInputContainer, submitBtn);

        submitBtn.style.display = 'block';
        submitBtn.disabled = false;
        submitBtn.textContent = 'Submit';

        renderPuzzleMedia(data);
    }

    function configureTextPuzzle(data) {
        if (inputGroup) {
            inputGroup.style.display = 'flex';
        }

        userAnswerInput.type = 'text';
        userAnswerInput.value = '';
        userAnswerInput.placeholder = 'Enter answer';

        submitBtn.style.display = 'block';
        submitBtn.disabled = false;
        submitBtn.textContent = 'Submit';

        renderPuzzleMedia(data);
    }

    function setupBingoSwap(data) {
        if (inputGroup) {
            inputGroup.style.display = 'none';
        }
        submitBtn.style.display = 'none';

        bingoSelectedCells = [];

        const gridSize = data.grid_size || [3, 3];
        const [rows, cols] = gridSize;

        const gridContainer = document.createElement('div');
        gridContainer.className = 'bingo-grid';
        gridContainer.style.display = 'grid';
        gridContainer.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
        gridContainer.style.gridTemplateRows = `repeat(${rows}, 1fr)`;
        gridContainer.style.gap = '4px';
        gridContainer.style.width = '100%';
        gridContainer.style.maxWidth = '640px';
        gridContainer.style.margin = '0 auto';

        const fullImg = new Image();
        fullImg.onload = () => {
            const cellWidth = fullImg.width / cols;
            const cellHeight = fullImg.height / rows;
            const totalCells = rows * cols;

            for (let i = 0; i < totalCells; i += 1) {
                const cell = document.createElement('div');
                cell.className = 'grid-cell';
                cell.dataset.index = i;
                cell.style.position = 'relative';
                cell.style.border = '2px solid #333';
                cell.style.borderRadius = '6px';
                cell.style.overflow = 'hidden';
                cell.style.cursor = 'pointer';
                cell.style.transition = 'transform 0.2s ease, border-color 0.2s ease';

                const cellImg = document.createElement('img');
                cellImg.className = 'cell-image';
                cellImg.style.width = '100%';
                cellImg.style.height = '100%';
                cellImg.style.objectFit = 'cover';
                cell.appendChild(cellImg);

                const canvas = document.createElement('canvas');
                canvas.width = cellWidth;
                canvas.height = cellHeight;
                const ctx = canvas.getContext('2d');
                const row = Math.floor(i / cols);
                const col = i % cols;
                ctx.drawImage(
                    fullImg,
                    col * cellWidth,
                    row * cellHeight,
                    cellWidth,
                    cellHeight,
                    0,
                    0,
                    cellWidth,
                    cellHeight
                );
                cellImg.src = canvas.toDataURL();

                const overlay = document.createElement('div');
                overlay.className = 'cell-overlay';
                overlay.style.position = 'absolute';
                overlay.style.top = '0';
                overlay.style.left = '0';
                overlay.style.width = '100%';
                overlay.style.height = '100%';
                overlay.style.backgroundColor = 'rgba(0, 120, 255, 0.5)';
                overlay.style.opacity = '0';
                overlay.style.transition = 'opacity 0.2s ease';
                overlay.style.pointerEvents = 'none';
                cell.appendChild(overlay);

                cell.addEventListener('click', () => toggleBingoCellSelection(i, cell));

                gridContainer.appendChild(cell);
            }

            puzzleImageContainer.appendChild(gridContainer);

            const submitSection = document.createElement('div');
            submitSection.className = 'bingo-submit';
            submitSection.style.textAlign = 'center';
            submitSection.style.marginTop = '18px';

            const bingoSubmitBtn = document.createElement('button');
            bingoSubmitBtn.textContent = 'Swap and Submit';
            bingoSubmitBtn.className = 'submit-bingo';
            bingoSubmitBtn.addEventListener('click', () => {
                if (bingoSelectedCells.length !== 2) {
                    showError('Please select exactly two cells to swap.');
                    return;
                }
                swapBingoCells();
                bingoSubmitBtn.disabled = true;
                bingoSubmitBtn.textContent = 'Processing...';
                submitAnswer();
            });

            submitSection.appendChild(bingoSubmitBtn);
            puzzleImageContainer.appendChild(submitSection);
        };

        fullImg.src = data.image_path;
    }

    function toggleBingoCellSelection(index, cellElement) {
        const overlay = cellElement.querySelector('.cell-overlay');

        const selectedIndex = bingoSelectedCells.indexOf(index);
        if (selectedIndex !== -1) {
            bingoSelectedCells.splice(selectedIndex, 1);
            logAction('cell_deselected', { puzzle_type: 'Bingo', cell_index: index });
            if (overlay) {
                overlay.style.opacity = '0';
            }
            cellElement.style.transform = 'scale(1)';
            cellElement.style.borderColor = '#333';
        } else {
            logAction('cell_selected', { puzzle_type: 'Bingo', cell_index: index });
            if (bingoSelectedCells.length === 2) {
                const firstIndex = bingoSelectedCells.shift();
                const firstCell = document.querySelector(`.grid-cell[data-index="${firstIndex}"]`);
                if (firstCell) {
                    const firstOverlay = firstCell.querySelector('.cell-overlay');
                    if (firstOverlay) {
                        firstOverlay.style.opacity = '0';
                    }
                    firstCell.style.transform = 'scale(1)';
                    firstCell.style.borderColor = '#333';
                }
            }

            bingoSelectedCells.push(index);
            if (overlay) {
                overlay.style.opacity = '1';
            }
            cellElement.style.transform = 'scale(0.96)';
            cellElement.style.borderColor = '#0078ff';
        }
    }

    function swapBingoCells() {
        if (bingoSelectedCells.length !== 2) {
            return;
        }

        const [firstIndex, secondIndex] = bingoSelectedCells;
        const firstCell = document.querySelector(`.grid-cell[data-index="${firstIndex}"]`);
        const secondCell = document.querySelector(`.grid-cell[data-index="${secondIndex}"]`);

        if (!firstCell || !secondCell) {
            return;
        }

        const firstImage = firstCell.querySelector('img');
        const secondImage = secondCell.querySelector('img');

        if (firstImage && secondImage) {
            const tempSrc = firstImage.src;
            firstImage.src = secondImage.src;
            secondImage.src = tempSrc;
        }
    }

    function setupShadowPlausibleGrid(data) {
        if (inputGroup) {
            inputGroup.style.display = 'none';
        }
        submitBtn.style.display = 'none';

        shadowSelectedCells = [];

        puzzleImageContainer.style.display = 'block';
        puzzleImageContainer.style.width = '100%';
        puzzleImageContainer.style.maxWidth = '1200px';
        puzzleImageContainer.style.margin = '0 auto';

        const gridContainer = document.createElement('div');
        gridContainer.className = 'shadow-plausible-grid';

        const optionImages = data.option_images || [];
        if (!optionImages.length) {
            showError('No shadow options available.');
            return;
        }

        const gridSize = data.grid_size || [];
        const cols = gridSize[1] || Math.ceil(Math.sqrt(optionImages.length));
        const rows = gridSize[0] || Math.ceil(optionImages.length / cols);
        // Use CSS grid-template-columns from stylesheet for better sizing
        gridContainer.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
        gridContainer.dataset.rows = rows;
        gridContainer.dataset.cols = cols;

        optionImages.forEach((src, index) => {
            const cell = document.createElement('div');
            cell.className = 'shadow-cell';
            cell.dataset.index = index;

            const img = document.createElement('img');
            img.src = src;
            img.alt = `Shadow option ${index + 1}`;
            img.draggable = false;
            cell.appendChild(img);

            const overlay = document.createElement('div');
            overlay.className = 'shadow-overlay';
            cell.appendChild(overlay);

            const checkmark = document.createElement('div');
            checkmark.className = 'shadow-checkmark';
            checkmark.textContent = '✓';
            cell.appendChild(checkmark);

            cell.addEventListener('click', () => toggleShadowSelection(index, cell));

            gridContainer.appendChild(cell);
        });

        puzzleImageContainer.appendChild(gridContainer);

        const submitSection = document.createElement('div');
        submitSection.className = 'shadow-submit';

        const shadowSubmitBtn = document.createElement('button');
        shadowSubmitBtn.textContent = 'Submit';
        shadowSubmitBtn.className = 'submit-shadow';
        shadowSubmitBtn.type = 'button';
        shadowSubmitBtn.addEventListener('click', () => {
            if (!shadowSelectedCells.length) {
                showError('Select at least one image before submitting.');
                return;
            }
            shadowSubmitBtn.disabled = true;
            shadowSubmitBtn.textContent = 'Processing...';
            submitAnswer();
        });

        submitSection.appendChild(shadowSubmitBtn);
        puzzleImageContainer.appendChild(submitSection);
    }

    function toggleShadowSelection(index, cellElement) {
        const overlay = cellElement.querySelector('.shadow-overlay');
        const checkmark = cellElement.querySelector('.shadow-checkmark');

        const alreadySelected = shadowSelectedCells.includes(index);
        if (alreadySelected) {
            shadowSelectedCells = shadowSelectedCells.filter((idx) => idx !== index);
            logAction('cell_deselected', { puzzle_type: 'Shadow_Plausible', cell_index: index });
            if (overlay) {
                overlay.style.opacity = '0';
            }
            if (checkmark) {
                checkmark.style.opacity = '0';
            }
            cellElement.style.transform = 'scale(1)';
            cellElement.style.borderColor = '#333';
        } else {
            shadowSelectedCells.push(index);
            logAction('cell_selected', { puzzle_type: 'Shadow_Plausible', cell_index: index });
            if (overlay) {
                overlay.style.opacity = '1';
            }
            if (checkmark) {
                checkmark.style.opacity = '1';
            }
            cellElement.style.transform = 'scale(0.97)';
            cellElement.style.borderColor = '#0078ff';
        }
    }

    function setupMirrorSelect(data) {
        if (inputGroup) {
            inputGroup.style.display = 'none';
        }
        submitBtn.style.display = 'none';

        mirrorSelectedCells = [];

        const layout = document.createElement('div');
        layout.className = 'mirror-layout';

        const referenceSection = document.createElement('div');
        referenceSection.className = 'mirror-reference';

        const referenceLabel = document.createElement('div');
        referenceLabel.className = 'mirror-reference-label';
        referenceLabel.textContent = 'Reference';
        referenceSection.appendChild(referenceLabel);

        const referenceImg = document.createElement('img');
        referenceImg.src = data.reference_image;
        referenceImg.alt = 'Reference object';
        referenceImg.draggable = false;
        referenceSection.appendChild(referenceImg);

        const optionsSection = document.createElement('div');
        optionsSection.className = 'mirror-options';

        const optionsLabel = document.createElement('div');
        optionsLabel.className = 'mirror-options-label';
        optionsLabel.textContent = 'Select all incorrect mirrors';
        optionsSection.appendChild(optionsLabel);

        const optionsGrid = document.createElement('div');
        optionsGrid.className = 'mirror-options-grid';

        const optionImages = data.option_images || [];
        if (!optionImages.length) {
            showError('No mirror options available.');
            return;
        }

        const gridSize = data.grid_size || [1, optionImages.length];
        const cols = gridSize[1] || optionImages.length || 1;
        optionsGrid.style.gridTemplateColumns = `repeat(${cols}, minmax(0, 1fr))`;

        optionImages.forEach((src, index) => {
            const cell = document.createElement('div');
            cell.className = 'mirror-option';
            cell.dataset.index = index;

            const img = document.createElement('img');
            img.src = src;
            img.alt = `Mirror option ${index + 1}`;
            img.draggable = false;
            cell.appendChild(img);

            const overlay = document.createElement('div');
            overlay.className = 'mirror-overlay';
            cell.appendChild(overlay);

            const badge = document.createElement('div');
            badge.className = 'mirror-checkmark';
            badge.textContent = '✕';
            cell.appendChild(badge);

            cell.addEventListener('click', () => toggleMirrorSelection(index, cell));

            optionsGrid.appendChild(cell);
        });

        optionsSection.appendChild(optionsGrid);
        layout.appendChild(referenceSection);
        layout.appendChild(optionsSection);
        
        // Reset container display for mirror layout
        puzzleImageContainer.style.display = 'block';
        puzzleImageContainer.appendChild(layout);

        const submitSection = document.createElement('div');
        submitSection.className = 'mirror-submit';

        const mirrorSubmitBtn = document.createElement('button');
        mirrorSubmitBtn.textContent = 'Submit';
        mirrorSubmitBtn.className = 'submit-mirror';
        mirrorSubmitBtn.type = 'button';
        mirrorSubmitBtn.addEventListener('click', () => {
            if (!mirrorSelectedCells.length) {
                showError('Select at least one mirror before submitting.');
                return;
            }
            mirrorSubmitBtn.disabled = true;
            mirrorSubmitBtn.textContent = 'Processing...';
            submitAnswer();
        });

        submitSection.appendChild(mirrorSubmitBtn);
        puzzleImageContainer.appendChild(submitSection);
    }

    function setupGridSelection(data) {
        if (inputGroup) {
            inputGroup.style.display = 'none';
        }
        submitBtn.style.display = 'none';

        // Make sure result message is visible (it's inside inputGroup)
        if (resultMessage) {
            resultMessage.style.display = 'block';
            resultMessage.style.position = 'relative';
            resultMessage.style.marginTop = '20px';
        }

        selectedGridCells = [];

        puzzleImageContainer.style.display = 'block';
        puzzleImageContainer.style.width = '100%';
        puzzleImageContainer.style.maxWidth = '960px';
        puzzleImageContainer.style.margin = '0 auto';

        // For Rhythm, show the reference GIF above the grid
        if (data.puzzle_type === 'Rhythm' && data.reference_gif) {
            const refContainer = document.createElement('div');
            refContainer.className = 'rhythm-reference';

            const refLabel = document.createElement('div');
            refLabel.className = 'rhythm-reference-label';
            refLabel.textContent = 'Reference Rhythm:';
            refContainer.appendChild(refLabel);

            const refImg = document.createElement('img');
            refImg.src = data.reference_gif;
            refImg.alt = 'Reference rhythm pattern';
            refContainer.appendChild(refImg);

            puzzleImageContainer.appendChild(refContainer);
        }

        // For Backmost_Layer, show the reference image above the grid
        if (data.puzzle_type === 'Backmost_Layer' && data.reference_image) {
            const refContainer = document.createElement('div');
            refContainer.className = 'backmost-reference';
            refContainer.style.textAlign = 'center';
            refContainer.style.marginBottom = '20px';

            const refLabel = document.createElement('div');
            refLabel.className = 'backmost-reference-label';
            refLabel.textContent = 'Reference (backmost shape to find):';
            refLabel.style.fontSize = '18px';
            refLabel.style.fontWeight = 'bold';
            refLabel.style.marginBottom = '10px';
            refContainer.appendChild(refLabel);

            const refImg = document.createElement('img');
            refImg.src = data.reference_image;
            refImg.alt = 'Reference pattern';
            refImg.style.width = '200px';
            refImg.style.height = '200px';
            refImg.style.border = '4px solid #0078ff';
            refImg.style.borderRadius = '8px';
            refContainer.appendChild(refImg);

            puzzleImageContainer.appendChild(refContainer);
        }

        // For Shadow_Direction, show the light direction arrow
        if (data.puzzle_type === 'Shadow_Direction' && data.reference_image) {
            const refContainer = document.createElement('div');
            refContainer.className = 'shadow-direction-reference';
            refContainer.style.textAlign = 'center';
            refContainer.style.marginBottom = '20px';

            const refLabel = document.createElement('div');
            refLabel.className = 'shadow-direction-reference-label';
            refLabel.textContent = 'Light Direction:';
            refLabel.style.fontSize = '18px';
            refLabel.style.fontWeight = 'bold';
            refLabel.style.marginBottom = '10px';
            refContainer.appendChild(refLabel);

            const refImg = document.createElement('img');
            refImg.src = data.reference_image;
            refImg.alt = 'Light direction arrow';
            refImg.style.width = '200px';
            refImg.style.height = '200px';
            refImg.style.border = '4px solid #ffcc00';
            refImg.style.borderRadius = '8px';
            refImg.style.backgroundColor = '#f5f5f0';
            refContainer.appendChild(refImg);

            puzzleImageContainer.appendChild(refContainer);
        }

        // For 3D_Viewpoint, show interactive 3D viewer or fallback to static image
        if (data.puzzle_type === '3D_Viewpoint') {
            // Clean up any existing viewer
            if (window.currentViewpointReference) {
                window.currentViewpointReference.dispose();
                window.currentViewpointReference = null;
            }

            if (data.shape_data) {
                // Use interactive Three.js viewer
                const viewerContainer = document.createElement('div');
                viewerContainer.className = 'viewpoint-viewer-container';
                viewerContainer.style.textAlign = 'center';
                viewerContainer.style.marginBottom = '20px';
                puzzleImageContainer.appendChild(viewerContainer);

                window.currentViewpointReference = new InteractiveViewpointReference(
                    viewerContainer,
                    data.shape_data
                );
            } else if (data.reference_image) {
                // Fallback to static image (for backwards compatibility)
                const refContainer = document.createElement('div');
                refContainer.className = 'viewpoint-reference';
                refContainer.style.textAlign = 'center';
                refContainer.style.marginBottom = '20px';

                const refLabel = document.createElement('div');
                refLabel.className = 'viewpoint-reference-label';
                refLabel.textContent = 'Reference 3D Object:';
                refLabel.style.fontSize = '18px';
                refLabel.style.fontWeight = 'bold';
                refLabel.style.marginBottom = '10px';
                refContainer.appendChild(refLabel);

                const refImg = document.createElement('img');
                refImg.src = data.reference_image;
                refImg.alt = '3D wireframe reference';
                refImg.style.width = '300px';
                refImg.style.height = '300px';
                refImg.style.border = '4px solid #0078ff';
                refImg.style.borderRadius = '8px';
                refImg.style.backgroundColor = '#1a1a2e';
                refContainer.appendChild(refImg);

                puzzleImageContainer.appendChild(refContainer);
            }
        }

        // For Box_Folding, show the unfolded cube net above the grid
        if (data.puzzle_type === 'Box_Folding' && data.reference_image) {
            const refContainer = document.createElement('div');
            refContainer.className = 'box-folding-reference';
            refContainer.style.textAlign = 'center';
            refContainer.style.marginBottom = '20px';

            const refLabel = document.createElement('div');
            refLabel.className = 'box-folding-reference-label';
            refLabel.textContent = 'Unfolded Cube Pattern:';
            refLabel.style.fontSize = '18px';
            refLabel.style.fontWeight = 'bold';
            refLabel.style.marginBottom = '10px';
            refContainer.appendChild(refLabel);

            const refImg = document.createElement('img');
            refImg.src = data.reference_image;
            refImg.alt = 'Unfolded cube net';
            refImg.style.maxWidth = '280px';
            refImg.style.width = '100%';
            refImg.style.border = '3px solid #0078ff';
            refImg.style.borderRadius = '6px';
            refImg.style.backgroundColor = '#ffffff';
            refContainer.appendChild(refImg);

            puzzleImageContainer.appendChild(refContainer);
        }

        // For Multi_Script, show the target characters above the grid
        if (data.puzzle_type === 'Multi_Script' && data.reference_image) {
            const refContainer = document.createElement('div');
            refContainer.className = 'multi-script-reference';
            refContainer.style.textAlign = 'center';
            refContainer.style.marginBottom = '20px';

            const refLabel = document.createElement('div');
            refLabel.className = 'multi-script-reference-label';
            refLabel.textContent = 'Target Characters (find these in the grid below):';
            refLabel.style.fontSize = '16px';
            refLabel.style.fontWeight = 'bold';
            refLabel.style.marginBottom = '8px';
            refContainer.appendChild(refLabel);

            const refHint = document.createElement('div');
            refHint.textContent = '(Characters in cells may be rotated or mirrored)';
            refHint.style.fontSize = '13px';
            refHint.style.color = '#666';
            refHint.style.marginBottom = '10px';
            refContainer.appendChild(refHint);

            const refImg = document.createElement('img');
            refImg.src = data.reference_image;
            refImg.alt = 'Target characters';
            refImg.style.maxWidth = '400px';
            refImg.style.width = '100%';
            refImg.style.border = '4px solid #0078ff';
            refImg.style.borderRadius = '8px';
            refImg.style.backgroundColor = '#ffffff';
            refContainer.appendChild(refImg);

            puzzleImageContainer.appendChild(refContainer);
        }

        // For Trajectory_Recovery, show the movement GIF above the grid
        if (data.puzzle_type === 'Trajectory_Recovery' && data.movement_gif) {
            const gifContainer = document.createElement('div');
            gifContainer.className = 'trajectory-gif-container';
            gifContainer.style.textAlign = 'center';
            gifContainer.style.marginBottom = '20px';

            const gifImg = document.createElement('img');
            gifImg.src = data.movement_gif;
            gifImg.alt = 'Ball movement trajectory';
            gifImg.style.maxWidth = '400px';
            gifImg.style.width = '100%';
            gifImg.style.border = '2px solid #333';
            gifImg.style.borderRadius = '8px';
            gifImg.draggable = false;

            gifContainer.appendChild(gifImg);
            puzzleImageContainer.appendChild(gifContainer);
        }

        // For Set_Game, show rules explanation above the grid
        if (data.puzzle_type === 'Set_Game') {
            const rulesContainer = document.createElement('div');
            rulesContainer.className = 'set-game-rules-container';
            rulesContainer.style.backgroundColor = '#f5f5f5';
            rulesContainer.style.border = '2px solid #333';
            rulesContainer.style.borderRadius = '8px';
            rulesContainer.style.padding = '15px';
            rulesContainer.style.marginBottom = '20px';
            rulesContainer.style.textAlign = 'left';

            const rulesTitle = document.createElement('h3');
            rulesTitle.textContent = 'Set Game Rules:';
            rulesTitle.style.marginTop = '0';
            rulesTitle.style.marginBottom = '10px';
            rulesTitle.style.fontSize = '18px';
            rulesTitle.style.fontWeight = 'bold';

            const rulesText = document.createElement('div');
            rulesText.innerHTML = `
                <p style="margin: 5px 0;"><strong>Each card has 4 attributes:</strong></p>
                <ul style="margin: 5px 0 5px 20px; padding: 0;">
                    <li><strong>Shape:</strong> circle, square, or triangle</li>
                    <li><strong>Color:</strong> red, green, or blue</li>
                    <li><strong>Count:</strong> 1, 2, or 3 shapes per card</li>
                    <li><strong>Fill:</strong> solid (filled), striped (lines), or empty (outline only)</li>
                </ul>
                <p style="margin: 5px 0;"><strong>How to find a valid Set:</strong> For each attribute, the 3 cards must be either <em>all the same</em> or <em>all different</em>.</p>
                <p style="margin: 5px 0; font-style: italic;">Select exactly 3 cards that match the condition shown in the prompt below.</p>
            `;
            rulesText.style.fontSize = '14px';
            rulesText.style.lineHeight = '1.5';

            rulesContainer.appendChild(rulesTitle);
            rulesContainer.appendChild(rulesText);
            puzzleImageContainer.appendChild(rulesContainer);
        }

        // For Global_Phase_Drift, show instructions
        if (data.puzzle_type === 'Global_Phase_Drift') {
            const instructionsContainer = document.createElement('div');
            instructionsContainer.className = 'global-phase-drift-instructions';
            instructionsContainer.style.textAlign = 'center';
            instructionsContainer.style.marginBottom = '20px';
            instructionsContainer.style.padding = '15px';
            instructionsContainer.style.backgroundColor = '#f8f8f8';
            instructionsContainer.style.border = '2px solid #666';
            instructionsContainer.style.borderRadius = '8px';

            const instructionsLabel = document.createElement('div');
            instructionsLabel.textContent = 'Watch the animations carefully - one cell is out of sync with the wave pattern';
            instructionsLabel.style.fontSize = '16px';
            instructionsLabel.style.fontWeight = 'bold';
            instructionsLabel.style.color = '#333';
            instructionsContainer.appendChild(instructionsLabel);

            puzzleImageContainer.appendChild(instructionsContainer);
        }

        // For Audio_Match, show audio player above the grid
        if (data.puzzle_type === 'Audio_Match' && data.audio_path) {
            const audioContainer = document.createElement('div');
            audioContainer.className = 'audio-match-container';
            audioContainer.style.textAlign = 'center';
            audioContainer.style.marginBottom = '20px';
            audioContainer.style.padding = '15px';
            audioContainer.style.backgroundColor = '#1a1a2e';
            audioContainer.style.border = '2px solid #00ff88';
            audioContainer.style.borderRadius = '12px';

            const audioLabel = document.createElement('div');
            audioLabel.textContent = 'Listen to the sound sequence:';
            audioLabel.style.fontSize = '18px';
            audioLabel.style.fontWeight = 'bold';
            audioLabel.style.color = '#00ff88';
            audioLabel.style.marginBottom = '15px';
            audioContainer.appendChild(audioLabel);

            const audio = document.createElement('audio');
            audio.src = data.audio_path;
            audio.controls = true;
            audio.autoplay = true;
            audio.style.width = '100%';
            audio.style.maxWidth = '400px';
            audioContainer.appendChild(audio);

            const replayBtn = document.createElement('button');
            replayBtn.textContent = '🔄 Replay Audio';
            replayBtn.style.cssText = 'margin-top: 12px; padding: 10px 20px; cursor: pointer; background: #00ff88; color: #1a1a2e; border: none; border-radius: 6px; font-weight: bold; font-size: 14px;';
            replayBtn.onclick = () => {
                audio.currentTime = 0;
                audio.play();
            };
            audioContainer.appendChild(replayBtn);

            const hint = document.createElement('div');
            hint.textContent = 'Select the cell showing the same sequence of sounds (single selection)';
            hint.style.marginTop = '12px';
            hint.style.fontSize = '14px';
            hint.style.color = '#aaa';
            audioContainer.appendChild(hint);

            puzzleImageContainer.appendChild(audioContainer);
        }

        const gridContainer = document.createElement('div');
        gridContainer.className = 'grid-container';

        // Add special class for Color_Counting to have white background
        if (data.puzzle_type === 'Color_Counting') {
            gridContainer.classList.add('color-counting-grid');
        }

        // Add special class for Hole_Counting for larger display with pixelated rendering
        if (data.puzzle_type === 'Hole_Counting') {
            gridContainer.classList.add('hole-counting-grid');
        }

        // Add special class for Trajectory_Recovery
        if (data.puzzle_type === 'Trajectory_Recovery') {
            gridContainer.classList.add('trajectory-recovery-grid');
        }

        // Add special class for Illusory_Ribbons for larger cells
        if (data.puzzle_type === 'Illusory_Ribbons') {
            gridContainer.classList.add('illusory-ribbons-grid');
        }

        // For Global_Phase_Drift, Temporal_Object_Continuity, and Structure_From_Motion, use cell_gifs instead of option_images
        const optionImages = (data.puzzle_type === 'Global_Phase_Drift' || data.puzzle_type === 'Temporal_Object_Continuity' || data.puzzle_type === 'Structure_From_Motion')
            ? (data.cell_gifs || [])
            : (data.option_images || []);

        if (!optionImages.length) {
            showError('No grid options available.');
            return;
        }

        const gridSize = data.grid_size || [3, 3];
        const cols = gridSize[1] || 3;
        const rows = gridSize[0] || 3;
        gridContainer.style.gridTemplateColumns = `repeat(${cols}, minmax(0, 1fr))`;
        gridContainer.dataset.rows = rows;
        gridContainer.dataset.cols = cols;

        optionImages.forEach((src, index) => {
            const cell = document.createElement('div');
            cell.className = 'grid-cell';
            cell.dataset.index = index;

            const img = document.createElement('img');
            img.src = src;
            img.alt = `Grid option ${index + 1}`;
            img.draggable = false;
            cell.appendChild(img);

            const overlay = document.createElement('div');
            overlay.className = 'grid-overlay';
            cell.appendChild(overlay);

            const checkmark = document.createElement('div');
            checkmark.className = 'grid-checkmark';
            checkmark.textContent = '✓';
            cell.appendChild(checkmark);

            cell.addEventListener('click', () => toggleGridSelection(index, cell));

            gridContainer.appendChild(cell);
        });

        puzzleImageContainer.appendChild(gridContainer);

        const submitSection = document.createElement('div');
        submitSection.className = 'grid-submit';

        const spookySubmitBtn = document.createElement('button');
        spookySubmitBtn.textContent = 'Submit';
        spookySubmitBtn.className = 'submit-grid';
        spookySubmitBtn.type = 'button';
        spookySubmitBtn.addEventListener('click', () => {
            if (!selectedGridCells.length) {
                showError('Select at least one cell before submitting.');
                return;
            }
            spookySubmitBtn.disabled = true;
            spookySubmitBtn.textContent = 'Processing...';
            submitAnswer();
        });

        submitSection.appendChild(spookySubmitBtn);
        puzzleImageContainer.appendChild(submitSection);
    }

    function toggleGridSelection(index, cellElement) {
        const overlay = cellElement.querySelector('.grid-overlay');
        const checkmark = cellElement.querySelector('.grid-checkmark');

        const alreadySelected = selectedGridCells.includes(index);
        if (alreadySelected) {
            selectedGridCells = selectedGridCells.filter((idx) => idx !== index);
            logAction('cell_deselected', { puzzle_type: currentPuzzle?.puzzle_type, cell_index: index });
            if (overlay) {
                overlay.style.opacity = '0';
            }
            if (checkmark) {
                checkmark.style.opacity = '0';
            }
            cellElement.style.transform = 'scale(1)';
            cellElement.style.borderColor = '#333';
        } else {
            selectedGridCells.push(index);
            logAction('cell_selected', { puzzle_type: currentPuzzle?.puzzle_type, cell_index: index });
            if (overlay) {
                overlay.style.opacity = '1';
            }
            if (checkmark) {
                checkmark.style.opacity = '1';
            }
            cellElement.style.transform = 'scale(0.97)';
            cellElement.style.borderColor = '#0078ff';
        }
    }


    function setupSquiggleSelect(data) {
        if (inputGroup) {
            inputGroup.style.display = 'none';
        }
        submitBtn.style.display = 'none';

        squiggleSelectedIndex = null;

        const optionImages = data.option_images || [];
        if (!optionImages.length) {
            showError('No squiggle options available.');
            return;
        }

        const revealDuration = Number.parseInt(data.reveal_duration, 10);
        const revealSeconds = Number.isFinite(revealDuration) && revealDuration > 0 ? revealDuration : 3;

        const previewWrapper = document.createElement('div');
        previewWrapper.className = 'squiggle-preview';

        const previewHint = document.createElement('div');
        previewHint.className = 'squiggle-hint';
        previewHint.textContent = `Memorize the trace. Choices appear in ${revealSeconds} second${revealSeconds === 1 ? '' : 's'}.`;
        previewWrapper.appendChild(previewHint);

        const previewImage = document.createElement('img');
        previewImage.src = data.reference_image;
        previewImage.alt = 'Trace preview';
        previewImage.draggable = false;
        previewImage.className = 'squiggle-preview-image';
        previewWrapper.appendChild(previewImage);

        puzzleImageContainer.appendChild(previewWrapper);

        const optionsGrid = document.createElement('div');
        optionsGrid.className = 'squiggle-options-grid';
        optionsGrid.style.display = 'none';

        // Reset container display for squiggle layout
        puzzleImageContainer.style.display = 'block';
        
        const gridSize = Array.isArray(data.grid_size) ? data.grid_size : null;
        if (gridSize && gridSize.length > 1 && Number.isFinite(gridSize[1]) && gridSize[1] > 0) {
            optionsGrid.style.gridTemplateColumns = `repeat(${gridSize[1]}, minmax(160px, 1fr))`;
        } else if (optionImages.length === 4) {
            optionsGrid.style.gridTemplateColumns = 'repeat(2, minmax(160px, 1fr))';
        }
        optionsGrid.style.columnGap = '40px';
        optionsGrid.style.rowGap = '32px';
        optionsGrid.style.justifyContent = 'center';

        optionImages.forEach((src, index) => {
            const option = document.createElement('div');
            option.className = 'squiggle-option';
            option.dataset.index = index;

            const img = document.createElement('img');
            img.src = src;
            img.alt = `Squiggle option ${index + 1}`;
            img.draggable = false;
            option.appendChild(img);

            option.addEventListener('click', () => selectSquiggleOption(index, option));

            optionsGrid.appendChild(option);
        });

        puzzleImageContainer.appendChild(optionsGrid);

        const submitSection = document.createElement('div');
        submitSection.className = 'squiggle-submit';
        submitSection.style.display = 'none';

        const squiggleSubmitBtn = document.createElement('button');
        squiggleSubmitBtn.className = 'submit-squiggle';
        squiggleSubmitBtn.type = 'button';
        squiggleSubmitBtn.textContent = 'Submit';
        squiggleSubmitBtn.addEventListener('click', () => {
            if (squiggleSelectedIndex === null) {
                showError('Select the squiggle that matches the preview.');
                return;
            }
            squiggleSubmitBtn.disabled = true;
            squiggleSubmitBtn.textContent = 'Processing...';
            submitAnswer();
        });

        submitSection.appendChild(squiggleSubmitBtn);
        puzzleImageContainer.appendChild(submitSection);

        squiggleRevealTimeout = setTimeout(() => {
            previewWrapper.remove();
            optionsGrid.style.display = 'grid';
            submitSection.style.display = 'flex';
        }, revealSeconds * 1000);
    }

    function selectSquiggleOption(index, optionElement) {
        if (squiggleSelectedIndex === index) {
            squiggleSelectedIndex = null;
            optionElement.classList.remove('active');
            logAction('option_deselected', { puzzle_type: 'Squiggle', option_index: index });
            return;
        }

        const previouslyActive = document.querySelector('.squiggle-option.active');
        if (previouslyActive) {
            previouslyActive.classList.remove('active');
        }

        squiggleSelectedIndex = index;
        optionElement.classList.add('active');
        logAction('option_selected', { puzzle_type: 'Squiggle', option_index: index });
    }

    function setupTransformPipelineSelect(data) {
        if (inputGroup) {
            inputGroup.style.display = 'none';
        }
        submitBtn.style.display = 'none';

        transformPipelineSelectedIndex = null;

        const optionImages = data.option_images || [];
        if (!optionImages.length) {
            showError('No transform pipeline options available.');
            return;
        }

        const container = document.createElement('div');
        container.className = 'transform-pipeline-container';

        // Reference image section
        const referenceSection = document.createElement('div');
        referenceSection.className = 'transform-pipeline-reference';

        const referenceLabel = document.createElement('div');
        referenceLabel.className = 'transform-pipeline-label';
        referenceLabel.textContent = 'Starting Image:';
        referenceSection.appendChild(referenceLabel);

        const referenceImage = document.createElement('img');
        referenceImage.src = data.reference_image;
        referenceImage.alt = 'Reference image';
        referenceImage.draggable = false;
        referenceImage.className = 'transform-pipeline-ref-image';
        referenceSection.appendChild(referenceImage);

        // Transform steps section
        const stepsSection = document.createElement('div');
        stepsSection.className = 'transform-pipeline-steps';

        const stepsLabel = document.createElement('div');
        stepsLabel.className = 'transform-pipeline-label';
        stepsLabel.textContent = 'Transform Steps:';
        stepsSection.appendChild(stepsLabel);

        const stepsList = document.createElement('div');
        stepsList.className = 'transform-pipeline-steps-list';
        (data.transform_steps || []).forEach((step, idx) => {
            const stepItem = document.createElement('div');
            stepItem.className = 'transform-pipeline-step';
            stepItem.textContent = `${idx + 1}. ${step}`;
            stepsList.appendChild(stepItem);
        });
        stepsSection.appendChild(stepsList);

        container.appendChild(referenceSection);
        container.appendChild(stepsSection);

        // Options grid
        const optionsGrid = document.createElement('div');
        optionsGrid.className = 'transform-pipeline-options-grid';

        const optionsLabel = document.createElement('div');
        optionsLabel.className = 'transform-pipeline-label';
        optionsLabel.textContent = 'Select the correct result:';
        optionsLabel.style.marginTop = '20px';
        container.appendChild(optionsLabel);

        puzzleImageContainer.style.display = 'block';

        const gridSize = Array.isArray(data.grid_size) ? data.grid_size : null;
        if (gridSize && gridSize.length > 1 && Number.isFinite(gridSize[1]) && gridSize[1] > 0) {
            optionsGrid.style.gridTemplateColumns = `repeat(${gridSize[1]}, minmax(160px, 1fr))`;
        } else if (optionImages.length === 4) {
            optionsGrid.style.gridTemplateColumns = 'repeat(2, minmax(160px, 1fr))';
        } else if (optionImages.length === 6) {
            optionsGrid.style.gridTemplateColumns = 'repeat(3, minmax(160px, 1fr))';
        }
        optionsGrid.style.columnGap = '40px';
        optionsGrid.style.rowGap = '32px';
        optionsGrid.style.justifyContent = 'center';
        optionsGrid.style.marginTop = '20px';

        optionImages.forEach((src, index) => {
            const option = document.createElement('div');
            option.className = 'transform-pipeline-option';
            option.dataset.index = index;

            const img = document.createElement('img');
            img.src = src;
            img.alt = `Option ${index + 1}`;
            img.draggable = false;
            option.appendChild(img);

            option.addEventListener('click', () => selectTransformPipelineOption(index, option));

            optionsGrid.appendChild(option);
        });

        container.appendChild(optionsGrid);

        // Submit button
        const submitSection = document.createElement('div');
        submitSection.className = 'transform-pipeline-submit';
        submitSection.style.display = 'flex';
        submitSection.style.justifyContent = 'center';
        submitSection.style.marginTop = '20px';

        const transformSubmitBtn = document.createElement('button');
        transformSubmitBtn.className = 'submit-transform-pipeline';
        transformSubmitBtn.type = 'button';
        transformSubmitBtn.textContent = 'Submit';
        transformSubmitBtn.addEventListener('click', () => {
            if (transformPipelineSelectedIndex === null) {
                showError('Select the correct transformed image.');
                return;
            }
            transformSubmitBtn.disabled = true;
            transformSubmitBtn.textContent = 'Processing...';
            submitAnswer();
        });

        submitSection.appendChild(transformSubmitBtn);
        container.appendChild(submitSection);

        puzzleImageContainer.appendChild(container);
    }

    function selectTransformPipelineOption(index, optionElement) {
        if (transformPipelineSelectedIndex === index) {
            transformPipelineSelectedIndex = null;
            optionElement.classList.remove('active');
            logAction('option_deselected', { puzzle_type: 'Transform_Pipeline', option_index: index });
            return;
        }

        const previouslyActive = document.querySelector('.transform-pipeline-option.active');
        if (previouslyActive) {
            previouslyActive.classList.remove('active');
        }

        transformPipelineSelectedIndex = index;
        optionElement.classList.add('active');
        logAction('option_selected', { puzzle_type: 'Transform_Pipeline', option_index: index });
    }

    function setupColorCipher(data) {
        const revealDuration = Number.parseInt(data.reveal_duration, 10);
        const revealSeconds = Number.isFinite(revealDuration) && revealDuration > 0 ? revealDuration : 3;

        if (inputGroup) {
            inputGroup.style.display = 'none';
        }

        submitBtn.style.display = 'block';
        submitBtn.disabled = true;
        submitBtn.textContent = 'Submit';

        userAnswerInput.type = data.input_mode === 'text' ? 'text' : 'number';
        userAnswerInput.value = '';
        userAnswerInput.placeholder = 'Enter answer';

        const previewWrapper = document.createElement('div');
        previewWrapper.className = 'color-cipher-preview';

        const previewTitle = document.createElement('div');
        previewTitle.className = 'color-cipher-title';
        previewTitle.textContent = 'Remember these values:';
        previewWrapper.appendChild(previewTitle);

        const mappingList = document.createElement('div');
        mappingList.className = 'color-cipher-mapping';

        (data.mapping || []).forEach((item) => {
            const row = document.createElement('div');
            row.className = 'color-cipher-row';

            const symbol = document.createElement('span');
            symbol.className = 'color-cipher-symbol';
            symbol.textContent = item.symbol || '';

            const value = document.createElement('span');
            value.className = 'color-cipher-value';
            value.textContent = `= ${item.value}`;

            row.appendChild(symbol);
            row.appendChild(value);
            mappingList.appendChild(row);
        });

        previewWrapper.appendChild(mappingList);
        puzzleImageContainer.appendChild(previewWrapper);

        const questionBlock = document.createElement('div');
        questionBlock.className = 'color-cipher-question';
        questionBlock.textContent = '';
        questionBlock.style.display = 'none';
        puzzleImageContainer.appendChild(questionBlock);

        colorCipherRevealTimeout = setTimeout(() => {
            previewWrapper.remove();
            if (inputGroup) {
                inputGroup.style.display = 'flex';
            }
            submitBtn.disabled = false;
            // questionBlock.textContent = data.question || 'What is the answer?';
            questionBlock.style.display = 'block';
            puzzlePrompt.textContent = data.question || 'What is the answer?';
            userAnswerInput.focus();
        }, revealSeconds * 1000);
    }

    function finalizeRedDotAttempt(answerPayload) {
        if (redDotAnswered) {
            return;
        }
        redDotAnswered = true;
        if (redDotTimeout) {
            clearTimeout(redDotTimeout);
            redDotTimeout = null;
        }

        if (redDotElement) {
            redDotElement.classList.add('red-dot-hidden');
        }

        const payload = {
            ...answerPayload,
            hit_index: redDotHits
        };

        submitRedDotAttempt(payload);
    }

    function toggleMirrorSelection(index, cellElement) {
        const overlay = cellElement.querySelector('.mirror-overlay');
        const badge = cellElement.querySelector('.mirror-checkmark');

        const alreadySelected = mirrorSelectedCells.includes(index);
        if (alreadySelected) {
            mirrorSelectedCells = mirrorSelectedCells.filter((idx) => idx !== index);
            logAction('cell_deselected', { puzzle_type: 'Mirror', cell_index: index });
            if (overlay) {
                overlay.classList.remove('active');
            }
            if (badge) {
                badge.classList.remove('active');
            }
            cellElement.classList.remove('active');
        } else {
            mirrorSelectedCells.push(index);
            logAction('cell_selected', { puzzle_type: 'Mirror', cell_index: index });
            if (overlay) {
                overlay.classList.add('active');
            }
            if (badge) {
                badge.classList.add('active');
            }
            cellElement.classList.add('active');
        }
    }

    function submitAnswer(overrideAnswer = undefined) {
        if (!currentPuzzle) {
            return;
        }

        if (currentPuzzle.input_type === 'red_dot_click') {
            return;
        }

        if ((currentPuzzle.input_type === 'number' || currentPuzzle.input_type === 'text') &&
            !userAnswerInput.value.trim()) {
            return;
        }

        const answerData = {
            puzzle_type: currentPuzzle.puzzle_type,
            puzzle_id: currentPuzzle.puzzle_id,
            session_id: sessionId
        };

        switch (currentPuzzle.input_type) {
            case 'number':
            case 'text':
                answerData.answer = userAnswerInput.value.trim();
                break;
            case 'dual_number':
                const input1 = document.getElementById('dual-number-input-1');
                const input2 = document.getElementById('dual-number-input-2');
                if (!input1.value || !input2.value) {
                    showError('Please enter both counts.');
                    resetCustomSubmitButtons();
                    return;
                }
                answerData.answer = `${input1.value},${input2.value}`;
                break;
            case 'bingo_swap':
                answerData.answer = bingoSelectedCells;
                if (bingoSelectedCells.length !== 2) {
                    showError('Please select exactly two cells to swap.');
                    resetCustomSubmitButtons();
                    return;
                }
                break;
            case 'shadow_plausible':
                answerData.answer = shadowSelectedCells;
                if (!shadowSelectedCells.length) {
                    showError('Select at least one image before submitting.');
                    resetCustomSubmitButtons();
                    return;
                }
                break;
            case 'mirror_select':
                answerData.answer = mirrorSelectedCells;
                if (!mirrorSelectedCells.length) {
                    showError('Select at least one mirror before submitting.');
                    resetCustomSubmitButtons();
                    return;
                }
                break;
            case 'squiggle_select':
                answerData.answer = squiggleSelectedIndex;
                if (squiggleSelectedIndex === null) {
                    showError('Select the squiggle that matches the preview.');
                    resetCustomSubmitButtons();
                    return;
                }
                break;
            case 'transform_pipeline_select':
                answerData.answer = transformPipelineSelectedIndex;
                if (transformPipelineSelectedIndex === null) {
                    showError('Select the correct transformed image.');
                    resetCustomSubmitButtons();
                    return;
                }
                break;
            case 'color_cipher':
                if (!userAnswerInput.value.trim()) {
                    showError('Enter your answer before submitting.');
                    resetCustomSubmitButtons();
                    return;
                }
                answerData.answer = userAnswerInput.value.trim();
                if (currentPuzzle.cipher_state) {
                    answerData.cipher_state = currentPuzzle.cipher_state;
                }
                break;
            case 'structure_from_motion_select':
            case 'circle_grid_select':
            case 'circle_grid_direction_select':
            case 'shape_grid_select':
            case 'color_counting_select':
            case 'hole_counting_select':
            case 'rotation_match_select':
            case 'rhythm_select':
            case 'backmost_layer_select':
            case 'shadow_direction_select':
            case 'global_phase_drift_select':
            case 'temporal_continuity_select':
            case 'layered_stack_select':
            case 'illusory_ribbons_select':
            case 'subway_paths_select':
            case 'trajectory_recovery_select':
            case 'set_game_select':
            case 'viewpoint_select':
            case 'box_folding_select':
            case 'illusion_grid_select':
            case 'multi_script_select':
                answerData.answer = selectedGridCells;
                if (!selectedGridCells.length) {
                    showError('Select at least one cell before submitting.');
                    resetCustomSubmitButtons();
                    return;
                }
                break;
            case 'audio_match_select':
                // Single selection - send first selected cell
                if (!selectedGridCells.length) {
                    showError('Select the cell matching the sound sequence.');
                    resetCustomSubmitButtons();
                    return;
                }
                answerData.answer = selectedGridCells[0];  // Single value
                break;
            case 'spooky_size_click':
                answerData.answer = spookySizeClickAnswer;
                break;
            case 'storyboard_logic':
                answerData.answer = storyboardOrder;
                if (!storyboardOrder || storyboardOrder.length === 0) {
                    showError('Please arrange the images before submitting.');
                    resetCustomSubmitButtons();
                    return;
                }
                break;
            case 'illusion_order':
                answerData.answer = illusionOrder;
                if (!illusionOrder || illusionOrder.length === 0) {
                    showError('Please arrange the images before submitting.');
                    resetCustomSubmitButtons();
                    return;
                }
                break;
            case 'jigsaw_puzzle':
                // Allow empty submissions - backend will mark as incorrect
                answerData.answer = jigsawPlacements || [];
                // Debug logging
                console.log('Jigsaw placements being submitted:', JSON.stringify(jigsawPlacements, null, 2));
                break;
            default:
                answerData.answer = userAnswerInput.value.trim();
                break;
        }

        answerData.elapsed_time = ((Date.now() - (puzzleStartTime || Date.now())) / 1000).toFixed(2);

        // Log submit action and add action sequence to answer data
        logAction('submit_answer', { answer: answerData.answer });
        answerData.action_sequence = actionSequence;

        if (submitBtn.style.display !== 'none') {
            submitBtn.disabled = true;
            submitBtn.textContent = 'Processing...';
        }

        fetch('/api/check_answer', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(answerData)
        })
            .then((response) => response.json())
            .then((data) => {
                if (data.error) {
                    throw new Error(data.error);
                }

                // Expose result to window for browser automation agents
                window.lastPuzzleResult = {
                    puzzle_type: currentPuzzle.puzzle_type,
                    puzzle_id: currentPuzzle.puzzle_id,
                    correct: data.correct,
                    correct_answer: data.correct_answer,
                    timestamp: Date.now()
                };

                // Debug logging for jigsaw puzzles
                if (currentPuzzle && currentPuzzle.input_type === 'jigsaw_puzzle') {
                    console.log('Jigsaw validation response:', data);
                    if (!data.correct && data.details) {
                        console.log('Validation details:', data.details);
                    }
                }

                benchmarkStats.total += 1;
                if (data.correct) {
                    benchmarkStats.correct += 1;
                    resultMessage.textContent = 'Correct!';
                    resultMessage.className = 'result-message correct';
                    createFireworks();
                } else {
                    resultMessage.textContent = 'Incorrect. Try again.';
                    resultMessage.className = 'result-message incorrect';
                    createSadFace();
                }

                updateStats();
                
                // Calculate cost per puzzle if agent cost data is available
                let cost_per_puzzle = null;
                if (window.__agentCostData && window.__agentCostData.averageCostPerPuzzle !== undefined) {
                    cost_per_puzzle = window.__agentCostData.averageCostPerPuzzle;
                } else if (window.__agentCostTracker && window.__agentCostTracker.puzzleCount > 0) {
                    // Fallback: calculate from tracker if available
                    cost_per_puzzle = window.__agentCostTracker.getAverageCost();
                }
                
                // Get model and provider metadata if available
                // First try window.__agentMetadata, then fallback to localStorage
                let model_name = null;
                let provider_name = null;
                let agent_framework = null;
                
                // Try to get from window first
                if (window.__agentMetadata) {
                    model_name = window.__agentMetadata.model || null;
                    provider_name = window.__agentMetadata.provider || null;
                    agent_framework = window.__agentMetadata.agentFramework || null;
                }
                
                // Fallback to localStorage if window doesn't have it
                if (!model_name && !provider_name) {
                    try {
                        const stored = localStorage.getItem('__agentMetadata');
                        if (stored) {
                            const metadata = JSON.parse(stored);
                            model_name = metadata.model || null;
                            provider_name = metadata.provider || null;
                            agent_framework = metadata.agentFramework || null;
                            // Also set it on window for future use
                            window.__agentMetadata = metadata;
                        }
                    } catch(e) {
                        console.warn('Could not read metadata from localStorage:', e);
                    }
                }
                
                // Debug logging (can be removed in production)
                if (model_name || provider_name) {
                    console.log('Recording benchmark result with metadata:', {
                        model: model_name,
                        provider: provider_name,
                        agent_framework: agent_framework,
                        source: window.__agentMetadata ? 'window' : 'localStorage'
                    });
                } else {
                    console.warn('No agent metadata available when recording benchmark result', {
                        hasWindowMetadata: !!window.__agentMetadata,
                        localStorageValue: localStorage.getItem('__agentMetadata')
                    });
                }
                
                recordBenchmarkResult({
                    puzzle_type: currentPuzzle.puzzle_type,
                    puzzle_id: currentPuzzle.puzzle_id,
                    user_answer: answerData.answer,
                    correct_answer: data.correct_answer,
                    correct: data.correct,
                    elapsed_time: data.elapsed_time || answerData.elapsed_time,
                    action_sequence: data.action_sequence || answerData.action_sequence,
                    ...(cost_per_puzzle !== null && { cost: cost_per_puzzle }),
                    // Don't include model/provider/agent_framework here - let recordBenchmarkResult add them from metadata
                });

                // Reset custom submit buttons (including jigsaw) before loading new puzzle
                resetCustomSubmitButtons();

                // Stop the timer since puzzle is complete
                stopTimer();

                setTimeout(loadNewPuzzle, 2000);
            })
            .catch((error) => {
                console.error('Error checking answer:', error);
                showError('Error checking answer. Please try again.');
                if (submitBtn.style.display !== 'none') {
                    submitBtn.disabled = false;
                    submitBtn.textContent = 'Submit';
                }
                resetCustomSubmitButtons();
            });
    }

    function resetCustomSubmitButtons() {
        const bingoButton = document.querySelector('.submit-bingo');
        if (bingoButton) {
            bingoButton.disabled = false;
            bingoButton.textContent = 'Swap and Submit';
        }

        const shadowButton = document.querySelector('.submit-shadow');
        if (shadowButton) {
            shadowButton.disabled = false;
            shadowButton.textContent = 'Submit';
        }

        const mirrorButton = document.querySelector('.submit-mirror');
        if (mirrorButton) {
            mirrorButton.disabled = false;
            mirrorButton.textContent = 'Submit';
        }

        const squiggleButton = document.querySelector('.submit-squiggle');
        if (squiggleButton) {
            squiggleButton.disabled = false;
            squiggleButton.textContent = 'Submit';
        }

        const transformPipelineButton = document.querySelector('.submit-transform-pipeline');
        if (transformPipelineButton) {
            transformPipelineButton.disabled = false;
            transformPipelineButton.textContent = 'Submit';
        }

        const storyboardButton = document.querySelector('.submit-storyboard');
        if (storyboardButton) {
            storyboardButton.disabled = false;
            storyboardButton.textContent = 'Submit Order';
        }

        const jigsawButton = document.querySelector('.submit-jigsaw');
        if (jigsawButton) {
            jigsawButton.disabled = false;
            jigsawButton.textContent = 'Submit Puzzle';
        }
    }

    function updateStats() {
        // Always update global stats (top row), regardless of active type
        totalCount.textContent = benchmarkStats.total;
        correctCount.textContent = benchmarkStats.correct;

        const accuracy = benchmarkStats.total
            ? ((benchmarkStats.correct / benchmarkStats.total) * 100).toFixed(1)
            : '0.0';
        accuracyEl.textContent = `${accuracy}%`;
    }

    function recordBenchmarkResult(result) {
        if (!result.timestamp) {
            result.timestamp = new Date().toISOString();
        }
        
        // Ensure cost is included if available from agent cost tracker
        if (result.cost === undefined && window.__agentCostData && window.__agentCostData.averageCostPerPuzzle !== undefined) {
            result.cost = window.__agentCostData.averageCostPerPuzzle;
        }
        
        // Ensure model/provider metadata is included if available from agent metadata
        // Check window first, then localStorage as fallback
        let metadata = window.__agentMetadata;
        if (!metadata || (!metadata.model && !metadata.provider)) {
            try {
                const stored = localStorage.getItem('__agentMetadata');
                if (stored) {
                    metadata = JSON.parse(stored);
                    window.__agentMetadata = metadata; // Cache it
                    console.log('Loaded metadata from localStorage:', metadata);
                }
            } catch(e) {
                console.warn('Could not read metadata from localStorage:', e);
            }
        }
        
        // Always set metadata if available (overwrite any existing values)
        if (metadata) {
            if (metadata.model) {
                result.model = metadata.model;
            }
            if (metadata.provider) {
                result.provider = metadata.provider;
            }
            if (metadata.agentFramework) {
                result.agent_framework = metadata.agentFramework;
            }
        } else {
            // No agent metadata - this is a human test
            result.model = result.model || 'human';
            result.provider = result.provider || 'human';
            result.agent_framework = result.agent_framework || 'human';
            console.log('Human test detected - setting model/provider/agent_framework to "human"');
        }

        // Debug: Log what we're sending
        console.log('Sending benchmark result:', JSON.stringify(result, null, 2));

        fetch('/api/benchmark_results', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(result)
        }).catch((error) => {
            console.error('Error recording benchmark result:', error);
        });
    }

    function displayDifficultyStars(puzzleType) {
        const difficultyRatings = {
            Dice_Count: 3,
            Bingo: 3,
            Shadow_Plausible: 4,
            Mirror: 4,
            Squiggle: 4,
            Color_Cipher: 3,
            Red_Dot: 4,
            Storyboard_Logic: 3,
            Static_Jigsaw: 2,
            Transform_Pipeline: 4,
        };

        const difficulty = difficultyRatings[puzzleType] || 1;
        const starsContainer = document.getElementById('difficulty-stars');
        if (!starsContainer) {
            return;
        }

        starsContainer.innerHTML = '';
        for (let i = 0; i < 5; i += 1) {
            const star = document.createElement('span');
            star.className = 'star';
            star.innerHTML = i < difficulty ? '★' : '☆';
            starsContainer.appendChild(star);
        }
    }

    function showError(message) {
        resultMessage.textContent = message;
        resultMessage.className = 'result-message incorrect';
    }

    function createFireworks() {
        const container = document.createElement('div');
        container.className = 'fireworks-container';

        const burstCount = 6;
        const sparkCount = 12;

        for (let burstIndex = 0; burstIndex < burstCount; burstIndex += 1) {
            const burst = document.createElement('div');
            burst.className = 'firework-burst';

            const topPercent = Math.random() * 70 + 10;
            const leftPercent = Math.random() * 80 + 10;
            burst.style.top = `${topPercent}%`;
            burst.style.left = `${leftPercent}%`;

            for (let sparkIndex = 0; sparkIndex < sparkCount; sparkIndex += 1) {
                const spark = document.createElement('span');
                spark.className = 'firework-spark';

                const hue = Math.floor(Math.random() * 360);
                spark.style.background = `radial-gradient(circle, hsl(${hue}, 100%, 70%) 0%, hsl(${hue}, 100%, 50%) 60%)`;

                spark.style.setProperty('--spark-index', sparkIndex);
                const delay = (burstIndex * 0.12) + (sparkIndex * 0.03);
                spark.style.animationDelay = `${delay}s`;

                burst.appendChild(spark);
            }

            container.appendChild(burst);
        }

        document.body.appendChild(container);

        setTimeout(() => {
            container.remove();
        }, 1600);
    }

    function createSadFace() {
        const container = document.createElement('div');
        container.className = 'sad-face-container';
        container.textContent = '😞';

        document.body.appendChild(container);

        setTimeout(() => {
            container.remove();
        }, 1500);
    }
});

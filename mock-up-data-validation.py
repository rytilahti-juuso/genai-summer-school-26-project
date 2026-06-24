from pathlib import Path
from pprint import pformat

records = [
    {
        "id": "http://arxiv.org/abs/2101.01472v1",
        "title": "Charge transport in layered nickelate heterostructures",
        "updated": "2021-01-05T11:24:18Z",
        "links": [
            "https://arxiv.org/abs/2101.01472v1",
            "https://arxiv.org/pdf/2101.01472v1",
        ],
        "summary": (
            "We investigate charge transport in layered nickelate heterostructures using "
            "a combination of density functional calculations and a minimal tight-binding "
            "model. The calculations reveal a strong dependence of the low-energy band "
            "structure on interface termination and epitaxial strain. We identify a regime "
            "in which orbital polarization enhances in-plane conductivity while suppressing "
            "interlayer hopping."
        ),
        "categories": ["cond-mat.mtrl-sci", "cond-mat.str-el"],
        "published": "2021-01-05T11:24:18Z",
        "comment": "12 pages, 6 figures",
        "primary_category": "cond-mat.mtrl-sci",
        "journal_ref": None,
        "authors": ["Elena V. Markovic", "Thomas R. Keller", "Junpei Sato"],
        "doi": None,
    },
    {
        "id": "http://arxiv.org/abs/2102.08311v2",
        "title": "Electron relaxation pathways in photoexcited transition-metal dichalcogenides",
        "updated": "2021-04-02T08:13:47Z",
        "links": [
            "https://arxiv.org/abs/2102.08311v2",
            "https://arxiv.org/pdf/2102.08311v2",
            "https://doi.org/10.1103/PhysRevB.104.045201",
        ],
        "summary": (
            "Time-resolved simulations are used to study electron relaxation in monolayer "
            "transition-metal dichalcogenides after ultrafast optical excitation. We show "
            "that intervalley scattering and coupling to optical phonons dominate the first "
            "few hundred femtoseconds. The predicted relaxation times agree with available "
            "pump-probe measurements over a broad temperature range."
        ),
        "categories": ["cond-mat.mes-hall", "physics.optics"],
        "published": "2021-02-16T17:41:09Z",
        "comment": "Accepted for publication in Physical Review B",
        "primary_category": "cond-mat.mes-hall",
        "journal_ref": "Phys. Rev. B 104, 045201 (2021)",
        "authors": ["Rina Okafor", "Mateo Alvarez", "Lars Nygaard"],
        "doi": "10.1103/PhysRevB.104.045201",
    },
    {
        "id": "http://arxiv.org/abs/2103.12690v1",
        "title": "A variational description of correlated electrons on frustrated lattices",
        "updated": "2021-03-23T14:56:31Z",
        "links": [
            "https://arxiv.org/abs/2103.12690v1",
            "https://arxiv.org/pdf/2103.12690v1",
        ],
        "summary": (
            "We introduce a variational wave-function ansatz for correlated electrons on "
            "frustrated lattices. The method combines tensor-network amplitudes with a "
            "symmetry-preserving Jastrow factor and can be optimized using stochastic "
            "gradients. Benchmarks on triangular and kagome clusters demonstrate improved "
            "ground-state energies relative to standard projected mean-field states."
        ),
        "categories": ["cond-mat.str-el", "quant-ph"],
        "published": "2021-03-23T14:56:31Z",
        "comment": None,
        "primary_category": "cond-mat.str-el",
        "journal_ref": None,
        "authors": ["Nadia Petrov", "Samuel D. Briggs"],
        "doi": None,
    },
    {
        "id": "http://arxiv.org/abs/2104.04438v3",
        "title": "Surface plasmon losses for electrons crossing nanostructured metallic films",
        "updated": "2021-09-18T10:22:04Z",
        "links": [
            "https://arxiv.org/abs/2104.04438v3",
            "https://arxiv.org/pdf/2104.04438v3",
            "https://doi.org/10.1016/j.elspec.2021.147129",
        ],
        "summary": (
            "The energy-loss spectrum of electrons traversing nanostructured metallic films "
            "is calculated within a nonlocal dielectric-response framework. Patterning the "
            "surface produces additional plasmon branches and redistributes spectral weight "
            "between bulk and surface modes. We discuss how these effects can be detected "
            "in momentum-resolved electron energy-loss spectroscopy."
        ),
        "categories": ["cond-mat.mtrl-sci", "physics.comp-ph"],
        "published": "2021-04-09T09:03:52Z",
        "comment": "18 pages, 8 figures; revised version",
        "primary_category": "cond-mat.mtrl-sci",
        "journal_ref": "J. Electron Spectrosc. Relat. Phenom. 254, 147129 (2021)",
        "authors": ["Irene Costa", "Pavel Morozov", "Hiroshi Tanaka"],
        "doi": "10.1016/j.elspec.2021.147129",
    },
    {
        "id": "http://arxiv.org/abs/2105.07142v1",
        "title": "Magnetic proximity effects in graphene on antiferromagnetic substrates",
        "updated": "2021-05-15T16:32:28Z",
        "links": [
            "https://arxiv.org/abs/2105.07142v1",
            "https://arxiv.org/pdf/2105.07142v1",
        ],
        "summary": (
            "First-principles calculations are employed to characterize magnetic proximity "
            "effects in graphene placed on several antiferromagnetic insulating substrates. "
            "Exchange splitting, sublattice asymmetry, and spin-orbit coupling are extracted "
            "from effective Hamiltonians. The results identify interface registries that "
            "support sizable spin splitting while retaining high carrier mobility."
        ),
        "categories": ["cond-mat.mes-hall", "physics.app-ph"],
        "published": "2021-05-15T16:32:28Z",
        "comment": "9 pages, 5 figures",
        "primary_category": "cond-mat.mes-hall",
        "journal_ref": None,
        "authors": ["Amira Bell", "Tomasz Kowalski", "Wei-Lun Chen"],
        "doi": None,
    },
    {
        "id": "http://arxiv.org/abs/2106.10255v2",
        "title": "Ultrafast control of orbital order in a correlated oxide",
        "updated": "2021-08-11T12:07:55Z",
        "links": [
            "https://arxiv.org/abs/2106.10255v2",
            "https://arxiv.org/pdf/2106.10255v2",
            "https://doi.org/10.1038/s41467-021-26544-8",
        ],
        "summary": (
            "We demonstrate theoretically that resonant mid-infrared excitation can control "
            "orbital order in a correlated transition-metal oxide. A driven lattice mode "
            "modifies the crystal-field splitting and triggers a transient redistribution "
            "of orbital occupation. The predicted response persists beyond the pump pulse "
            "and provides a route toward nonthermal switching of electronic phases."
        ),
        "categories": ["cond-mat.str-el", "cond-mat.mtrl-sci"],
        "published": "2021-06-18T13:48:02Z",
        "comment": "Supplementary material included",
        "primary_category": "cond-mat.str-el",
        "journal_ref": "Nat. Commun. 12, 6120 (2021)",
        "authors": ["Lucia Ferraro", "Daniel K. Price", "Arun Mehta", "Sofia Lindholm"],
        "doi": "10.1038/s41467-021-26544-8",
    },
    {
        "id": "http://arxiv.org/abs/2107.01984v1",
        "title": "Electron-phonon coupling in twisted bilayer semiconductors",
        "updated": "2021-07-05T07:39:11Z",
        "links": [
            "https://arxiv.org/abs/2107.01984v1",
            "https://arxiv.org/pdf/2107.01984v1",
        ],
        "summary": (
            "We derive a continuum model for electron-phonon coupling in twisted bilayer "
            "semiconductors. Moire reconstruction produces spatially modulated coupling "
            "vertices and strongly momentum-dependent scattering rates. Numerical results "
            "show that acoustic phonons can substantially limit mobility near isolated "
            "moire minibands."
        ),
        "categories": ["cond-mat.mes-hall"],
        "published": "2021-07-05T07:39:11Z",
        "comment": None,
        "primary_category": "cond-mat.mes-hall",
        "journal_ref": None,
        "authors": ["Kevin J. Rhodes", "Yuna Park"],
        "doi": None,
    },
    {
        "id": "http://arxiv.org/abs/2108.09317v2",
        "title": "Topological edge states in patterned photonic graphene",
        "updated": "2021-11-02T15:18:43Z",
        "links": [
            "https://arxiv.org/abs/2108.09317v2",
            "https://arxiv.org/pdf/2108.09317v2",
            "https://doi.org/10.1103/PhysRevA.105.013515",
        ],
        "summary": (
            "We study topological edge states in a photonic analogue of graphene formed by "
            "a patterned dielectric slab. Breaking inversion symmetry opens a valley gap, "
            "while domain walls host robust guided modes. Full-wave simulations demonstrate "
            "low-loss propagation around sharp bends and quantify the role of fabrication "
            "disorder."
        ),
        "categories": ["physics.optics", "cond-mat.mes-hall"],
        "published": "2021-08-20T19:26:34Z",
        "comment": "14 pages, 7 figures",
        "primary_category": "physics.optics",
        "journal_ref": "Phys. Rev. A 105, 013515 (2022)",
        "authors": ["Mina Haddad", "Oliver Stein", "Grace M. Liu"],
        "doi": "10.1103/PhysRevA.105.013515",
    },
    {
        "id": "http://arxiv.org/abs/2109.05061v1",
        "title": "Spin-resolved electron scattering from two-dimensional magnets",
        "updated": "2021-09-10T10:11:26Z",
        "links": [
            "https://arxiv.org/abs/2109.05061v1",
            "https://arxiv.org/pdf/2109.05061v1",
        ],
        "summary": (
            "A scattering formalism is developed for spin-polarized electrons incident on "
            "two-dimensional magnetic crystals. The theory includes exchange, spin-orbit "
            "coupling, and inelastic magnon emission. Calculations for representative "
            "ferromagnetic monolayers predict pronounced spin asymmetries that can serve "
            "as signatures of collective magnetic excitations."
        ),
        "categories": ["cond-mat.mtrl-sci", "physics.atom-ph"],
        "published": "2021-09-10T10:11:26Z",
        "comment": "10 pages, 4 figures",
        "primary_category": "cond-mat.mtrl-sci",
        "journal_ref": None,
        "authors": ["Peter N. Wallace", "Aya Nakamura", "Mikkel Sorensen"],
        "doi": None,
    },
    {
        "id": "http://arxiv.org/abs/2110.13624v2",
        "title": "Nonequilibrium electron dynamics in a driven Hubbard chain",
        "updated": "2022-01-14T13:49:05Z",
        "links": [
            "https://arxiv.org/abs/2110.13624v2",
            "https://arxiv.org/pdf/2110.13624v2",
            "https://doi.org/10.1103/PhysRevResearch.4.023091",
        ],
        "summary": (
            "We analyze nonequilibrium electron dynamics in a periodically driven Hubbard "
            "chain using time-dependent matrix-product states. The drive generates doublon-"
            "holon pairs and induces a crossover from coherent oscillations to diffusive "
            "energy absorption. Finite-size scaling reveals a broad prethermal regime at "
            "high driving frequencies."
        ),
        "categories": ["cond-mat.str-el", "quant-ph"],
        "published": "2021-10-26T18:04:47Z",
        "comment": "Revised manuscript with additional finite-size analysis",
        "primary_category": "cond-mat.str-el",
        "journal_ref": "Phys. Rev. Research 4, 023091 (2022)",
        "authors": ["Jonas Richter", "Priya Deshmukh", "Caleb Norton"],
        "doi": "10.1103/PhysRevResearch.4.023091",
    },
    {
        "id": "http://arxiv.org/abs/2111.07731v1",
        "title": "Electronic structure of vacancy-ordered perovskite oxides",
        "updated": "2021-11-15T09:57:38Z",
        "links": [
            "https://arxiv.org/abs/2111.07731v1",
            "https://arxiv.org/pdf/2111.07731v1",
        ],
        "summary": (
            "The electronic structure of vacancy-ordered perovskite oxides is investigated "
            "with hybrid density functional theory. Oxygen-vacancy ordering produces narrow "
            "transition-metal bands and several competing magnetic configurations. We map "
            "the resulting low-energy states to an effective multiorbital model and discuss "
            "implications for transport and catalysis."
        ),
        "categories": ["cond-mat.mtrl-sci"],
        "published": "2021-11-15T09:57:38Z",
        "comment": None,
        "primary_category": "cond-mat.mtrl-sci",
        "journal_ref": None,
        "authors": ["Fatima Rahman", "Ethan Cole", "Noah Ben-Ami"],
        "doi": None,
    },
    {
        "id": "http://arxiv.org/abs/2112.11802v3",
        "title": "High-harmonic generation from correlated electron systems",
        "updated": "2022-05-19T06:44:12Z",
        "links": [
            "https://arxiv.org/abs/2112.11802v3",
            "https://arxiv.org/pdf/2112.11802v3",
            "https://doi.org/10.1103/PhysRevLett.129.017401",
        ],
        "summary": (
            "High-harmonic generation in correlated electron systems is studied using "
            "nonequilibrium dynamical mean-field theory. Strong interactions reshape the "
            "harmonic plateau and introduce features associated with doublon recombination. "
            "We identify scaling relations between the cutoff energy, the field amplitude, "
            "and the interaction strength."
        ),
        "categories": ["cond-mat.str-el", "physics.optics"],
        "published": "2021-12-22T20:08:14Z",
        "comment": "5 pages, 4 figures; supplemental material",
        "primary_category": "cond-mat.str-el",
        "journal_ref": "Phys. Rev. Lett. 129, 017401 (2022)",
        "authors": ["Katarina Vogel", "Miguel Santos", "Robert J. Ellis"],
        "doi": "10.1103/PhysRevLett.129.017401",
    },
    {
        "id": "http://arxiv.org/abs/2201.03216v1",
        "title": "Electron beam shaping with programmable electrostatic phase plates",
        "updated": "2022-01-10T15:12:59Z",
        "links": [
            "https://arxiv.org/abs/2201.03216v1",
            "https://arxiv.org/pdf/2201.03216v1",
        ],
        "summary": (
            "We propose a programmable electrostatic phase plate for shaping free-electron "
            "wave functions in transmission electron microscopes. Independent electrodes "
            "enable dynamic control over phase profiles and allow the generation of vortex, "
            "Bessel, and aberration-corrected beams. Simulations quantify efficiency and "
            "robustness under realistic fabrication constraints."
        ),
        "categories": ["physics.ins-det", "physics.optics"],
        "published": "2022-01-10T15:12:59Z",
        "comment": "11 pages, 6 figures",
        "primary_category": "physics.ins-det",
        "journal_ref": None,
        "authors": ["Leonie Hartmann", "Victor S. Ramos"],
        "doi": None,
    },
    {
        "id": "http://arxiv.org/abs/2202.06743v2",
        "title": "Quantum oscillations in a compensated semimetal under pressure",
        "updated": "2022-04-27T11:31:42Z",
        "links": [
            "https://arxiv.org/abs/2202.06743v2",
            "https://arxiv.org/pdf/2202.06743v2",
            "https://doi.org/10.1103/PhysRevMaterials.6.064202",
        ],
        "summary": (
            "Magnetotransport measurements and first-principles calculations are combined "
            "to investigate quantum oscillations in a compensated semimetal under pressure. "
            "The oscillation spectrum reveals a pressure-induced Lifshitz transition and a "
            "substantial enhancement of the cyclotron mass. The results establish a direct "
            "connection between Fermi-surface reconstruction and anomalous magnetoresistance."
        ),
        "categories": ["cond-mat.mtrl-sci", "cond-mat.mes-hall"],
        "published": "2022-02-14T18:20:05Z",
        "comment": "16 pages, 9 figures",
        "primary_category": "cond-mat.mtrl-sci",
        "journal_ref": "Phys. Rev. Materials 6, 064202 (2022)",
        "authors": ["Helena Ortiz", "Marcus Field", "Sang-Min Lee", "Claire Dupont"],
        "doi": "10.1103/PhysRevMaterials.6.064202",
    },
    {
        "id": "http://arxiv.org/abs/2203.10955v1",
        "title": "Excitonic signatures in time-resolved electron energy-loss spectroscopy",
        "updated": "2022-03-21T12:45:16Z",
        "links": [
            "https://arxiv.org/abs/2203.10955v1",
            "https://arxiv.org/pdf/2203.10955v1",
        ],
        "summary": (
            "We calculate time-resolved electron energy-loss spectra for a model "
            "semiconductor following resonant excitation. Bound excitons produce transient "
            "peaks below the equilibrium band gap, while ionized carriers broaden the "
            "continuum response. The results provide practical criteria for separating "
            "excitonic and free-carrier contributions in ultrafast measurements."
        ),
        "categories": ["cond-mat.mtrl-sci", "physics.optics"],
        "published": "2022-03-21T12:45:16Z",
        "comment": None,
        "primary_category": "cond-mat.mtrl-sci",
        "journal_ref": None,
        "authors": ["Isabel Moreno", "Dae-Hyun Kim", "George P. Foster"],
        "doi": None,
    },
    {
        "id": "http://arxiv.org/abs/2204.15408v2",
        "title": "Spin-orbit torque in metallic antiferromagnet thin films",
        "updated": "2022-07-09T14:33:27Z",
        "links": [
            "https://arxiv.org/abs/2204.15408v2",
            "https://arxiv.org/pdf/2204.15408v2",
            "https://doi.org/10.1063/5.0102381",
        ],
        "summary": (
            "We study spin-orbit torque in metallic antiferromagnet thin films using a "
            "Boltzmann transport treatment informed by first-principles band structures. "
            "Crystal symmetry determines the allowed torque components and leads to a large "
            "staggered response for selected film orientations. We propose experimental "
            "geometries for distinguishing field-like and damping-like contributions."
        ),
        "categories": ["cond-mat.mes-hall", "physics.app-ph"],
        "published": "2022-04-29T17:52:40Z",
        "comment": "Accepted in Applied Physics Letters",
        "primary_category": "cond-mat.mes-hall",
        "journal_ref": "Appl. Phys. Lett. 121, 062402 (2022)",
        "authors": ["Anika Sharma", "Paul E. Jensen", "Koji Matsuda"],
        "doi": "10.1063/5.0102381",
    },
    {
        "id": "http://arxiv.org/abs/2205.08829v1",
        "title": "Electronic reconstruction at polar oxide interfaces",
        "updated": "2022-05-18T08:06:33Z",
        "links": [
            "https://arxiv.org/abs/2205.08829v1",
            "https://arxiv.org/pdf/2205.08829v1",
        ],
        "summary": (
            "A self-consistent electrostatic and tight-binding model is used to study "
            "electronic reconstruction at polar oxide interfaces. The calculations capture "
            "charge transfer, dielectric screening, and confinement across multiple atomic "
            "layers. We identify threshold thicknesses for interfacial conduction and show "
            "how cation intermixing modifies the carrier distribution."
        ),
        "categories": ["cond-mat.mtrl-sci", "cond-mat.str-el"],
        "published": "2022-05-18T08:06:33Z",
        "comment": "13 pages, 7 figures",
        "primary_category": "cond-mat.mtrl-sci",
        "journal_ref": None,
        "authors": ["Camille Laurent", "Rohit Bansal"],
        "doi": None,
    },
    {
        "id": "http://arxiv.org/abs/2206.12177v2",
        "title": "Room-temperature electron mobility in halide perovskites from first principles",
        "updated": "2022-10-03T16:21:51Z",
        "links": [
            "https://arxiv.org/abs/2206.12177v2",
            "https://arxiv.org/pdf/2206.12177v2",
            "https://doi.org/10.1021/acs.jpclett.2c02571",
        ],
        "summary": (
            "We compute room-temperature electron mobility in halide perovskites using "
            "first-principles electron-phonon matrix elements and iterative solutions of the "
            "Boltzmann equation. Soft polar modes dominate scattering, but dynamic screening "
            "substantially reduces their impact. The calculated mobilities reproduce measured "
            "trends across several compositions."
        ),
        "categories": ["cond-mat.mtrl-sci"],
        "published": "2022-06-24T10:15:08Z",
        "comment": "Additional convergence tests added",
        "primary_category": "cond-mat.mtrl-sci",
        "journal_ref": "J. Phys. Chem. Lett. 13, 9214 (2022)",
        "authors": ["Sarah K. Patel", "Nikolai Werner", "Mei Qiao"],
        "doi": "10.1021/acs.jpclett.2c02571",
    },
    {
        "id": "http://arxiv.org/abs/2207.14318v1",
        "title": "Correlated electron pairing in flat-band kagome metals",
        "updated": "2022-07-29T13:17:22Z",
        "links": [
            "https://arxiv.org/abs/2207.14318v1",
            "https://arxiv.org/pdf/2207.14318v1",
        ],
        "summary": (
            "We examine pairing instabilities in a multiorbital model of flat-band kagome "
            "metals. Functional renormalization-group calculations reveal competition "
            "between unconventional superconductivity and charge order. The dominant pairing "
            "symmetry changes with filling and interaction range, producing several distinct "
            "regions in the phase diagram."
        ),
        "categories": ["cond-mat.supr-con", "cond-mat.str-el"],
        "published": "2022-07-29T13:17:22Z",
        "comment": None,
        "primary_category": "cond-mat.supr-con",
        "journal_ref": None,
        "authors": ["David M. Klein", "Xiaoyu Ren", "Beatriz Oliveira"],
        "doi": None,
    },
    {
        "id": "http://arxiv.org/abs/2208.07466v3",
        "title": "Electron tomography of nanoscale strain fields in semiconductor devices",
        "updated": "2023-01-12T09:54:36Z",
        "links": [
            "https://arxiv.org/abs/2208.07466v3",
            "https://arxiv.org/pdf/2208.07466v3",
            "https://doi.org/10.1038/s41565-023-01288-6",
        ],
        "summary": (
            "We present an electron-tomography method for reconstructing three-dimensional "
            "strain fields in nanoscale semiconductor devices. Diffraction patterns acquired "
            "over multiple tilt angles are inverted with a regularized phase-retrieval "
            "algorithm. Tests on simulated and experimental structures demonstrate nanometer "
            "spatial resolution and sensitivity to sub-percent lattice distortions."
        ),
        "categories": ["physics.ins-det", "cond-mat.mtrl-sci"],
        "published": "2022-08-15T14:28:19Z",
        "comment": "Main text and supplementary information",
        "primary_category": "physics.ins-det",
        "journal_ref": "Nat. Nanotechnol. 18, 412 (2023)",
        "authors": ["Emma J. Rowe", "Felix Baumann", "Akira Ito", "Laura Chen"],
        "doi": "10.1038/s41565-023-01288-6",
    },
    {
        "id": "http://arxiv.org/abs/2209.11604v1",
        "title": "Nonlinear electron hydrodynamics in narrow graphene channels",
        "updated": "2022-09-23T07:46:10Z",
        "links": [
            "https://arxiv.org/abs/2209.11604v1",
            "https://arxiv.org/pdf/2209.11604v1",
        ],
        "summary": (
            "Nonlinear electron hydrodynamics in narrow graphene channels is investigated "
            "using numerical solutions of the Navier-Stokes equations coupled to electrostatic "
            "screening. At high current density, vortices shift toward the channel boundaries "
            "and generate a measurable second-harmonic voltage. The predicted signatures "
            "distinguish viscous flow from ballistic transport."
        ),
        "categories": ["cond-mat.mes-hall"],
        "published": "2022-09-23T07:46:10Z",
        "comment": "8 pages, 5 figures",
        "primary_category": "cond-mat.mes-hall",
        "journal_ref": None,
        "authors": ["Oskar Lind", "Jasmine Brooks"],
        "doi": None,
    },
    {
        "id": "http://arxiv.org/abs/2210.09153v2",
        "title": "Pressure-tuned magnetism in a layered van der Waals crystal",
        "updated": "2023-02-06T18:12:48Z",
        "links": [
            "https://arxiv.org/abs/2210.09153v2",
            "https://arxiv.org/pdf/2210.09153v2",
            "https://doi.org/10.1103/PhysRevB.107.104421",
        ],
        "summary": (
            "We combine Raman spectroscopy, transport measurements, and density functional "
            "theory to study pressure-tuned magnetism in a layered van der Waals crystal. "
            "Compression enhances interlayer exchange and drives a transition from "
            "antiferromagnetic to ferromagnetic stacking. The transition is accompanied by "
            "a pronounced reconstruction of the electronic bands."
        ),
        "categories": ["cond-mat.mtrl-sci", "cond-mat.mes-hall"],
        "published": "2022-10-17T15:05:29Z",
        "comment": "Revised after peer review",
        "primary_category": "cond-mat.mtrl-sci",
        "journal_ref": "Phys. Rev. B 107, 104421 (2023)",
        "authors": ["Marta Zielinska", "Connor Hayes", "Liang Zhou"],
        "doi": "10.1103/PhysRevB.107.104421",
    },
    {
        "id": "http://arxiv.org/abs/2211.13820v1",
        "title": "Electron correlation effects in molecular junction conductance",
        "updated": "2022-11-24T12:37:44Z",
        "links": [
            "https://arxiv.org/abs/2211.13820v1",
            "https://arxiv.org/pdf/2211.13820v1",
        ],
        "summary": (
            "We assess electron correlation effects in molecular junction conductance using "
            "a combination of many-body perturbation theory and nonequilibrium Green "
            "functions. Dynamical screening shifts frontier molecular levels and reduces "
            "the zero-bias conductance compared with semilocal density functional results. "
            "The magnitude of the correction depends strongly on electrode separation."
        ),
        "categories": ["cond-mat.mes-hall", "physics.chem-ph"],
        "published": "2022-11-24T12:37:44Z",
        "comment": None,
        "primary_category": "cond-mat.mes-hall",
        "journal_ref": None,
        "authors": ["Nora Svensson", "Alexei Grigoriev", "Daniela Ruiz"],
        "doi": None,
    },
    {
        "id": "http://arxiv.org/abs/2212.06491v2",
        "title": "Machine-learned potentials for electron irradiation damage in silicon",
        "updated": "2023-04-14T10:02:17Z",
        "links": [
            "https://arxiv.org/abs/2212.06491v2",
            "https://arxiv.org/pdf/2212.06491v2",
            "https://doi.org/10.1088/2632-2153/accf01",
        ],
        "summary": (
            "Machine-learned interatomic potentials are developed to simulate electron "
            "irradiation damage in crystalline silicon. The training data include displaced "
            "configurations, defect structures, and high-energy collision trajectories. "
            "Large-scale simulations reproduce threshold displacement energies and reveal "
            "the early stages of defect-cluster formation."
        ),
        "categories": ["cond-mat.mtrl-sci", "cs.LG"],
        "published": "2022-12-13T09:28:51Z",
        "comment": "Code and training data are available with the publication",
        "primary_category": "cond-mat.mtrl-sci",
        "journal_ref": "Mach. Learn.: Sci. Technol. 4, 025017 (2023)",
        "authors": ["Gregor Weiss", "Lina Adebayo", "Martin E. Cook"],
        "doi": "10.1088/2632-2153/accf01",
    },
    {
        "id": "http://arxiv.org/abs/2301.09742v1",
        "title": "Collective electron modes in moire superlattices",
        "updated": "2023-01-23T17:34:06Z",
        "links": [
            "https://arxiv.org/abs/2301.09742v1",
            "https://arxiv.org/pdf/2301.09742v1",
        ],
        "summary": (
            "We investigate collective electron modes in moire superlattices within the "
            "random-phase approximation. Narrow minibands support low-energy plasmons whose "
            "dispersion is highly sensitive to filling and twist angle. Landau damping is "
            "suppressed in selected density windows, suggesting favorable conditions for "
            "long-lived charge oscillations."
        ),
        "categories": ["cond-mat.mes-hall", "cond-mat.str-el"],
        "published": "2023-01-23T17:34:06Z",
        "comment": "15 pages, 8 figures",
        "primary_category": "cond-mat.mes-hall",
        "journal_ref": None,
        "authors": ["Sergio Marin", "Hannah Voss", "Takeshi Mori"],
        "doi": None,
    },
]

assert len(records) == 25
required = {
    "id", "title", "updated", "links", "summary", "categories", "published",
    "comment", "primary_category", "journal_ref", "authors", "doi"
}
assert all(set(record) == required for record in records)

content = (
    "from typing import Any\n\n"
    "MOCK_ARXIV_RECORDS: list[dict[str, Any]] = "
    + pformat(records, width=100, sort_dicts=False)
    + "\n"
)

output_path = Path("/mnt/data/mock_arxiv_records_25.py")
output_path.write_text(content, encoding="utf-8")
print(f"Created {output_path} with {len(records)} records.")

pub struct KineticMC {
    pub temp_K: f64,
    pub kb: f64,
    pub reaction_time: f64,
    pub P_H2: f64,
    pub m_size: usize,
    pub params: Parameters,
    pub chain_length: usize,
    pub carbon_array: Vec<u8>,
    pub chain_array: Vec<u8>,
    pub hydrogen_array: Vec<u8>,
    pub current_time: f64,
    chains: Option<Vec<(usize, usize)>>,
    chains_valid: bool,
}

pub struct Parameters {
    pub k_ads: f64,
    pub A_des: f64,
    pub E0_des: f64,
    pub alpha_vdw: f64,
    pub A_scission: f64,
    pub E_dMC: f64,
    pub E_scission: f64,
    pub beta_int: f64,
    pub K_H2: f64,
}

pub enum ReactionType {
    Desorption,
    Adsorption,
    Dmc,
    Scission,
}

//constructing Parameters struct 
impl Parameters {
    pub fn default() -> Self {
        Self {
            k_ads: 1e-2,
            A_des: 1e-3,
            E0_des: 0.5,
            alpha_vdw: 0.05,
            A_scission: 1e-3,
            E_dMC: 1.0,
            E_scission: 1.2,
            beta_int: 0.2,
            K_H2: 1e-2,
        }
    }
}

impl KineticMC {
    pub fn new(
        temp_C: f64,
        reaction_time: f64,
        chain_length: Option<usize>,
        params: Option<Parameters>,
        P_H2: f64,
        m_size: usize,
    ) -> Self {
        let temp_K = temp_C + 273.15;
        let kb = 1.380649e-23;  // Boltzmann constant in J/K
        
        let params = params.unwrap_or_else(|| Parameters::default());
        
        let chain_length = chain_length.unwrap_or_else(|| {
            //single chain length sampling
            Self::sample_normal_distr(280.0, 10.0) 
        });
        
        let mut kmc = Self {
            temp_K,
            kb,
            reaction_time,
            P_H2,
            m_size,
            params,
            chain_length,
            carbon_array: Vec::new(),
            chain_array: Vec::new(),
            hydrogen_array: Vec::new(),
            current_time: 0.0,
            chains: None,
            chains_valid: false,
        };
        
        kmc.init_arrays(chain_length);
        kmc
    }
    
    fn init_arrays(&mut self, chain_length: usize) {
        // Carbon array: all free initially
        self.carbon_array = vec![0; chain_length];
        
        // Chain array: internal bonds = 1, boundaries = 0
        self.chain_array = vec![0; chain_length + 1];
        for i in 1..chain_length {
            self.chain_array[i] = 1;
        }
        
        // Hydrogen array: internal=2, terminals=3
        self.hydrogen_array = vec![2; chain_length];
        self.hydrogen_array[0] = 3;
        if chain_length > 1 {
            self.hydrogen_array[chain_length - 1] = 3;
        }
    }

    pub fn get_rate(&self, n: usize, reaction_type: ReactionType, is_internal: bool) -> f64 {
        const R_EV: f64 = 8.617e-5;  // Boltzmann constant in eV/K
        let t = self.temp_K;
        let p = &self.params;
        
        let penalty = if is_internal { p.beta_int } else { 0.0 };
        
        match reaction_type {
            ReactionType::Desorption => {
                let e_total = p.E0_des + (p.alpha_vdw * n as f64);
                p.A_des * (-e_total / (R_EV * t)).exp()
            }
            ReactionType::Adsorption => {
                let theta_h = (p.K_H2 * self.P_H2).sqrt() / (1.0 + (p.K_H2 * self.P_H2).sqrt());
                let available_sites = 1.0 - theta_h; //might be overkill, if we track H coverage explicitly
                p.k_ads * available_sites
            }
            ReactionType::Dmc => {
                let e_act = p.E_dMC + penalty;
                p.A_scission * (-e_act / (R_EV * t)).exp()
            }
            ReactionType::Scission => {
                let e_act = p.E_scission + penalty;
                p.A_scission * (-e_act / (R_EV * t)).exp()
            }
        }
}
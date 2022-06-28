


class Tracker():

    def __init__(self, number_of_iterations, termination_eps, warp_mode):
        self.number_of_iterations = number_of_iterations
        self.termination_eps = termination_eps
        self.warp_mode = warp_mode

        self.frame_number = 0
        self.last_image = None
        self.tracks = []
        #self.results = 

    def _compute_warp_matrix(self, img, last_image, number_of_iterations, termination_eps, warp_mode):
        # return warp_matrix
        pass

    #def 

    def add(self, new_id_map):
        pass

    def forward(self, img, bboxes, scores):
        
        # Motion compensation
        if self.frame_number > 1:
            warp_matrix = self._compute_warp_matrix(img, self.last_image, self.number_of_iterations, self.termination_eps, self.warp_mode)
        
        # Track matching
        matched_map = None
        matched_mask = np.zeros(img.shape[:-1]).astype(np.int16)
        if len(self.tracks) > 0:
            # Search for a match or update
        else:
            # Initialize tracks
            init_ids = list(range(1, len(bboxes) + 1))
            new_id_map = {n_id : n_pos for (n_id, n_pos) in zip(init_ids, bboxes)}
            matched_map = new_id_map
        
        # Update tracking data
        self.add(new_id_map)

        # Generate outputs
        for t in self.tracks:
            if t.id not in self.results.keys():
                self.results[t.id] = {}
            position = t.pos
            self.results[t.id][self.frame_number] = np.concatenate([position, np.array([1.])])
        self.last_image = img

        return self.results, matched_map # matched_map in orther to generate and the current processed frame outside
    
    __call__ = forward

    def get_results(self):
        return self.results
    

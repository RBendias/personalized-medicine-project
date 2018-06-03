
        
    
        ####fbpca
        pipeline = Pipeline([('tfidf', TfidfVectorizer(preprocessor=preprocessor, tokenizer=tokenizer,            
                                                          max_features= max_features , max_df= max_df, 
                                                          min_df=min_df, ngram_range=(1,ngram), stop_words=stopWords)) 
                           ])
           
        train = pipeline.fit_transform(self.X_train)
        test = pipeline.transform(self.X_test)
        stage2_transf = pipeline.transform(self.stage2)
        stack = vstack([train,  test,stage2_transf ]).toarray()
        (U, s, Va) = pca(stack, k=200, raw=False, n_iter=50, l=None)
        transformed_set = np.dot(U,np.diag(s))
        
        train = transformed_set[:self.shape_train[0]]
        test = transformed_set[self.shape_train[0]: self.shape_train[0] + self.shape_test[0]]
        stage2_transf = transformed_set[self.shape_train[0]+self.shape_test[0]:self.shape_train[0]+self.shape_test[0]+self.shape_stage2test[0]]
        
        return train, test, stage2_transf, pipeline
    
       ### sklearn_pca
        from sklearn.decomposition import PCA
        pipeline = Pipeline([('tfidf', TfidfVectorizer(preprocessor=preprocessor, tokenizer=tokenizer,            
                                                          max_features= max_features , max_df= max_df, 
                                                          min_df=min_df, ngram_range=(1,ngram), stop_words=stopWords)), 
                            ('svd', PCA(n_components = n_comp, n_iter = n_iter)),
                            ('norm', normalizer)])
            train, test, stage2_transf = self.prep(pipeline, self.X_train,  self.X_test, self.stage2)
            
            
            
           
        
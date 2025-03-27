class HealthInsurance:

    def __init__( self ):
        self.home_path = ''
        self.annual_premium_scaler =            pickle.load( open( self.home_path + 'features/encoding_annual_premium.pkl', 'rb' ) )
        self.age_scaler =                       pickle.load( open( self.home_path + 'features/encoding_age.pkl', 'rb' ) )
        self.vintage_scaler =                   pickle.load( open( self.home_path + 'features/encoding_vintage.pkl', 'rb' ) )
        self.target_encode_gender_scaler =      pickle.load( open( self.home_path + 'features/encoding_gender.pkl', 'rb' ) )
        self.target_encode_region_code_scaler = pickle.load( open( self.home_path + 'features/encoding_region_code.pkl', 'rb' ) )
        self.fe_policy_sales_channel_scaler =   pickle.load( open( self.home_path + 'features/encoding_policy_sales_channel.pkl', 'rb' ) )
    
    def feature_engineering(self, df):
        # Create Copy for this session
        df2 = df.copy()
        
        # Vehicle Damage
        df2['Vehicle_Damage'] = df2['Vehicle_Damage'].apply( lambda x: 1 if x == 'Yes' else 0 )
        
        # Vehicle Age
        df2['Vehicle_Age'] =  df2['Vehicle_Age'].apply( lambda x: 'over_2_years' if x == '> 2 Years' else 'between_1_2_year' if x == '1-2 Year' else 'below_1_year' )
        
        return df2

    
    
    def data_preparation(self, df2):
        # annual premium - Standard Scaler
        df2['Annual_Premium'] = self.annual_premium_scaler.transform(df2[['Annual_Premium']].values)
                                                                     
        # Age - MinMax Scaler
        df2['Age'] = self.age_scaller.transform(df2[['Age']].values)

        # Vintage - MinMax Scaler
        df2['Vintage'] = self.vintage_scaler.transform(df2[['Vintage']].values)

        # Gender - One Hot Encoding / Target Encoding
        df2.loc[:, 'Gender'] = df2['Gender'].map(self.target_encode_gender_scaler)
        
        # Region_Code - Frequency Encoding / Target Encoding
        df2.loc[:, 'Region_Code'] = df2['Region_Code'].map(self.target_encode_region_code_scaler)
        
        # Vehicle_Age - One Hot Encoding / Frequency Encoding
        df2 = pd.get_dummies(df2, prefix='Vehicle_Age', columns=['Vehicle_Age'])
        
        # Policy_Sales_Channel - Frequency Encoding / Target Encoding
        df2.loc[:, 'Policy_Sales_Channel'] = df2['Policy_Sales_Channel'].map(self.fe_policy_sales_channel_scaler)

        # Selecting the most relevant features
        cols_selected = ['Annual_Premium',
                         'Vintage',
                         'Age',
                         'Region_Code',
                         'Vehicle_Damage',
                         'Previously_Insured',
                         'Policy_Sales_Channel']

        return df2[cols_selected]


    
    def get_prediction( self, model, original_data, test_data ):
        # model prediction
        pred = model.predict_proba( test_data )

        # join prediction into original data
        original_data['Score'] = pred

        return original_data.to_json( orient='records', date_format='iso' )
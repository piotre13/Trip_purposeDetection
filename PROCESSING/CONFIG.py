


categories =['home','work','eating','entertainment','recreation',
				'shopping','travel','admin_chores','religious','health','police','education','commuting']



TYPES = {

'home':['lodging'],

'work' :['accounting','electrician','insurance_agency','lawyer',
			'locksmith','moving_company','painter','plumber','real_estate_agency',
			'roofing_contractor','storage'],

'eating':['food','bar','cafe','meal_delivery','meal_takeaway','restaurant'],

'entertainment' :['amusement_park','acquarium','art_gallery','bowling_alley',
					'casino','movie_rental','movie_theater','museum','night_club',
					'stadium','tourist_attraction','zoo'],

'recreation':['beauty_salon','beauty_salon','campground','gym','hair_care',
				'jewelry_store','laundry','park','rv_park','spa','travel_agency'],

'shopping' :['bakery','bicycle_store','book_store','car_dealer','car_rental',
				'car_repair','car_wash','clothing_store','convenience_store',
				'department_store','drugstore','electronics_store','florist',
				'furniture_store','grocery_or_supermarket','hardware_store','home_goods_store',
				'liquor_store','pet_store','shoe_store','shopping_mall','store','supermarket','tematic_market',
				'market'],

'travel' :['airport','transit_station','move_out_of_the_city'],

'admin_chores' :['administrative','atm','bank','city_hall','courthouse','embassy',
							'local_government_office','parking','post_office','taking_kids_to_schoo'],

'religious' :['religion','church','cemetery','hindu_temple','mosque','synagogue'],

'health' :['health','dentist','doctor','hospital','pharmacy','physiotherapist',
				'veterinary_care'],

'police' :['fire_station','police'],

'education' :['primary_school','school','secondary_school','university','library'],

'commuting':['bus_station','light_rail_station','subway_station',
				'taxi_stand','train_station','transit_station']

}


trip_format = {
				'_id': 0,
				'user_id': 0,
				'O':
						{
							'lat': 0,
							'lng':0,
							'timestamp':0,
							'census_zone':0

						},
				'D':
						{
							'lat': 0,
							'lng':0,
							'timestamp':0,
							'census_zone':0

						},

				'activity_time':0,
				'category':
						{
							'home':0,
							'work':0,
							'eating':0,
							'entertainment':0,
							'recreation':0,
							'shopping':0,
							'travel':0,
							'admin_chores':0,
							'religious':0,
							'health':0,
							'police':0,
							'education':0

						},
						
				'mode':0,
				}











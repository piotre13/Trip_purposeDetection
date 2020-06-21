from telegram.ext import Updater, CommandHandler, MessageHandler
from telegram.ext import ConversationHandler, Filters
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove
import googlemaps
import logging
from FirebaseClass import Firebase
from BotFilters import *
from config import GOOGLEKEY as google_key
from config import BOTKEY as bot_key
import sys
import pandas as pd
import pickle
import numpy as np
from scipy.stats import planck

sys.path.insert(1, '../')
from CensusMatching import CensusMatching

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO)

logger = logging.getLogger(__name__)


class TorinoBot:
    def __init__(self):
        self.threshold_n_trips = 100
        self.new_trips = 0
        self.new_user = 0
        self.modes = ['unknown_activity_type', 'Car', 'Walk', 'Bike',
                      'Bus/Tram', 'in_passenger_vehicle', 'Train', 'in_bus', 'Subway',
                      'flying', 'motorcycling', 'running']

        self.fb = Firebase()
        self.fb.authenticate()
        self.df = pd.read_excel('data/df.xlsx', index_col=0)
        self.RF_model = pickle.load(open('data/rf.sav', 'rb'))
        self.territorial_info = pd.read_excel('data/FINAL_territorial.xlsx', index_col=1).drop(columns=['Unnamed: 0'])
        self.territorial_features = [
            'P_TOT', 'MALE_TOT', 'FEM_TOT', 'age 0-9', 'age 10-24', 'age 25-39', 'age 40-64', 'age >65', 'male 0-9',
            'male 10-24', 'male 25-39',
            'male 40-64', 'male >65', 'P47', 'P48', 'P49', 'P50', 'P51', 'P52', 'P61', 'P62', 'P128', 'P130', 'P131',
            'P135', 'P137', 'P138',
            'ST1', 'ST3', 'ST4', 'ST5', 'ST9', 'ST10', 'ST11', 'ST12', 'ST13', 'PF1', 'PF2', 'PF3', 'PF4', 'PF5', 'PF6',
            'PF7',
            'PF8', 'PF9', 'INCOMEinco', 'home', 'work', 'eating', 'entertainment', 'recreation', 'shopping', 'travel',
            'admin_chores', 'religious',
            'health', 'police', 'education'
        ]

        self.columns = ['activity_time', 'mode', 'P_TOT', 'MALE_TOT', 'FEM_TOT',
                        'age 0-9', 'age 10-24', 'age 25-39', 'age 40-64', 'age >65', 'male 0-9',
                        'male 10-24', 'male 25-39', 'male 40-64', 'male >65', 'P47', 'P48',
                        'P49', 'P50', 'P51', 'P52', 'P61', 'P62', 'P128', 'P130', 'P131',
                        'P135', 'P137', 'P138', 'ST1', 'ST3', 'ST4', 'ST5', 'ST9', 'ST10',
                        'ST11', 'ST12', 'ST13', 'PF1', 'PF2', 'PF3', 'PF4', 'PF5', 'PF6', 'PF7',
                        'PF8', 'PF9', 'INCOMEinco', 'home', 'work', 'eating', 'entertainment',
                        'recreation', 'shopping', 'travel', 'admin_chores', 'religious',
                        'health', 'police', 'education', 'age', 'gender', 'occupation',
                        'd_hour', 'bin_weekday']

        self.n = 12
        self.genders = [['Male', 'Female']]
        self.occupations = [['Student', 'Worker', 'Retired']]
        self.mode_of_transportation = [['Walk', 'Bike', 'Car'], ['Bus/Tram', 'Train', 'Subway'], ['Other']]
        self.activity_time = [
            ['10 min', '10-20 min', '20-30 min'],
            ['30-60 min', '1-2 h', '2-3 h'],
            ['3-8 h', '>8 h']
        ]
        self.categories = [
            ['Home', 'Work', 'Eating', 'Entertainment'],
            ['Recreation', 'Shopping', 'Travel', 'Chores'],
            ['Religious', 'Health', 'Police', 'Education', 'Commuting']
        ]

        self.age_filter = AgeFilter()
        self.mode_filter = ModeFilter()
        self.gender_filter = GenderFilter()
        self.activity_filter = ActivityTimeFilter()
        self.game_filter = GameModeFilter()
        self.occupation_filter = OccupationFilter()
        self.destination_start_filter = DestinationStartFilter()
        self.is_week_filter = WeekFilter()
        self.category_filter = CategoryFilter()

        self.cm = CensusMatching()
        self.gmaps = googlemaps.Client(key=google_key)

        self.START, self.CHOOSE, self.AGE, self.GND, self.OCC, self.ORG, self.DEST, self.MODE, self.DESTINATION_START, self.ACT, self.IS_WEEK, self.PRE_GAME, self.GAME, self.CHECK, self.PRP, self.SAVE = range(
            16)
        self.STATE = self.START
        logger.info('BOT RUNNING')

    def start(self, bot, update):
        self.user = str(bot.message.from_user.id)
        logger.info(f'STARTED by: {self.user}')
        welcome = f'(0/{self.n}) - Welcome to TorinoBot. This bot tries to guess the purpose of a trip that you took. ' \
                  f'Please follow the instructions providing the information needed. '
        help_ = 'Use /help to check the possible commands'
        bot.message.reply_text(welcome)
        bot.message.reply_text(help_)
        fb_users = self.fb.download('users')[1]
        check = any(self.user in x for x in fb_users)
        if check:
            keyboard = [['Statistics', 'New Trip']]
            reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
            bot.message.reply_text('Choose any of the following', reply_markup=reply_markup)

            return self.CHOOSE
        else:
            self.new_user = 1
            message = f'(1/{self.n}) - Insert your Age'
            bot.message.reply_text(message)

            return self.AGE

    def choose(self, bot, update):
        if bot.message.text == 'Statistics':
            diary, n_trips, most_category, most_destination = self.statistical_analysis()
            if n_trips > 0:
                bot.message.reply_text(diary)
                bot.message.reply_text(f"Number of Trips: {n_trips}")
                bot.message.reply_text(f"Most Frequent Purpose: {most_category[0]}")
                bot.message.reply_text(f"Most Frequent Destination {most_destination[0]}")
                message = 'Choose any of the following'
            else:
                message = "You don't have any trips saved. Add a new trip"

            keyboard = [['Statistics', 'New Trip']]
            reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
            bot.message.reply_text(message, reply_markup=reply_markup)

            return self.CHOOSE
        else:
            message = f'(5/{self.n}) - Insert the origin of your trip. Be as specific as possible (insert the city at ' \
                      f'the end e.g) '
            bot.message.reply_text(message)

            return self.ORG

    def age(self, bot, update):
        self.age = int(bot.message.text)
        logger.info(f'{self.user}-->{self.age}')
        message = f'({self.AGE + 1}/{self.n}) - Insert your gender: M/F?'
        reply_markup = ReplyKeyboardMarkup(self.genders, one_time_keyboard=True, resize_keyboard=True)
        bot.message.reply_text(message, reply_markup=reply_markup)

        return self.GND

    def gender(self, bot, update):
        gnd = bot.message.text
        if gnd == 'Male' or gnd == 'M':
            self.gender = 0
        elif gnd == 'Female' or gnd == 'F':
            self.gender = 1
        logger.info(f'{self.user}-->{self.gender}')
        message = f'({self.GND + 1}/{self.n}) - Insert your occupation from the list'
        reply_markup = ReplyKeyboardMarkup(self.occupations, one_time_keyboard=True, resize_keyboard=True)
        bot.message.reply_text(message, reply_markup=reply_markup)

        return self.OCC

    def occupation(self, bot, update):
        self.occupation = self.occupations[0].index(bot.message.text)
        logger.info(f'{self.user}-->{self.occupation}')
        self.save_user()
        message = f'({self.OCC + 1}/{self.n}) - Insert the Origin of your trip. Please be more specific as possible (' \
                  f'use the address/name and then the city) '
        bot.message.reply_text(message)

        return self.ORG

    def save_user(self):
        tmp_obj = {
            "age": self.age,
            "gender": self.gender,
            "occupation": self.occupation
        }
        self.fb.upload('users', tmp_obj, key=self.user)
        logger.info(tmp_obj)
        logger.info(f'{self.user}-->UPLOADED')

    def find_address(self, address):
        logger.info('Started looking for address')
        geometry_location = self.gmaps.find_place(address, 'textquery',
                                                  fields=['geometry/location', 'name', 'formatted_address'])
        logger.info(geometry_location)

        return geometry_location

    def origin(self, bot, update):
        address = bot.message.text
        logger.info(f'ORIGIN: {self.user}-->{address}')
        dic = self.find_address(address)
        self.o_lat = dic['candidates'][0]['geometry']['location']['lat']
        self.o_lng = dic['candidates'][0]['geometry']['location']['lng']
        self.origin_name = dic['candidates'][0]['name']
        self.origin_address = dic['candidates'][0]['formatted_address']
        self.o_census_id = self.cm.census_matching(self.o_lat, self.o_lng)
        message = f'({self.ORG + 1}/{self.n}) - Insert now the Destination of your trip. Please be more specific as ' \
                  f'possible (use the address/name and then the city) '
        origin = f'From {self.origin_address}'
        bot.message.reply_text(origin)
        bot.message.reply_text(message)

        return self.DEST

    def destination(self, bot, update):
        address = bot.message.text
        logger.info(f'DESTINATION: {self.user}-->{address}')
        dic = self.find_address(address)
        self.d_lat = dic['candidates'][0]['geometry']['location']['lat']
        self.d_lng = dic['candidates'][0]['geometry']['location']['lng']
        self.destination_name = dic['candidates'][0]['name']
        self.destination_address = dic['candidates'][0]['formatted_address']
        self.d_census_id = self.cm.census_matching(self.d_lat, self.d_lng)
        dest = f'To {self.destination_address}'
        bot.message.reply_text(dest)
        message = f'({self.DEST + 1}/{self.n}) - Origin-Destination saved. Insert now your mode of transportation ' \
                  f'from the list '
        reply_markup = ReplyKeyboardMarkup(self.mode_of_transportation, one_time_keyboard=True, resize_keyboard=True)
        bot.message.reply_text(message, reply_markup=reply_markup)
        logger.info('MODE')

        return self.MODE

    def mode(self, bot, update):
        self.mode = self.modes.index(bot.message.text)
        logger.info(f'{self.user}-->{self.mode}')
        message = f'({self.MODE + 1}/{self.n}) - At what time did you arrive at the destination? (Type just the Hour)'
        bot.message.reply_text(message)

        return self.DESTINATION_START

    def destination_start(self, bot, update):
        self.finish_time = int(bot.message.text) // 8
        logger.info(f'{self.user}-->{self.finish_time}')
        message = f'({self.DESTINATION_START + 1}/{self.n}) - Insert now the activity time of your trip (how much ' \
                  f'time did you spend on the destination) '
        reply_markup = ReplyKeyboardMarkup(self.activity_time, one_time_keyboard=True, resize_keyboard=True)
        bot.message.reply_text(message, reply_markup=reply_markup)

        return self.ACT

    def activity(self, bot, update):
        self.activity = bot.message.text
        x = [(i, a.index(self.activity)) for i, a in enumerate(self.activity_time) if self.activity in a]
        x_index = 3 * x[0][0] + x[0][1]
        self.activity = x_index

        logger.info(f'{self.user}-->{self.activity}')
        message = f'({self.ACT + 1}/{self.n} - Was a week-trip or a weekend-trip?'
        reply_markup = ReplyKeyboardMarkup([['WeekDay Trip', 'WeekEnd Trip']])
        bot.message.reply_text(message, reply_markup=reply_markup)

        return self.IS_WEEK

    def is_week(self, bot, update):
        if bot.message.text in ['0', 0, 'weekend', 'weekend-trip']:
            self.week = 0
        else:
            self.week = 1

        logger.info(f'{self.user}-->{self.week}')
        message = f'({self.IS_WEEK + 1}/{self.n} - Want to see if the MachineLearningAlgorithm guess the purpose?'
        keyboard = [['Yes', 'No']]
        reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)
        bot.message.reply_text(message, reply_markup=reply_markup)

        return self.PRE_GAME

    def pre_game(self, bot, update):
        is_game = bot.message.text
        if is_game == 'Yes':
            new_data = self.create_dataset(self.user)
            p = self.RF_model.predict(new_data)[0]
            self.prediction = p.upper()

            message = f'({self.PRE_GAME + 1}/{self.n} - Your Trip Purpose is: *drum roll*\n{self.prediction}'
            bot.message.reply_text(message)
            message = 'Is it correct? If not, tap on the correct trip purpose'
            reply_markup = ReplyKeyboardMarkup(self.categories, one_time_keyboard=True, resize_keyboard=True)
            bot.message.reply_text(message, reply_markup=reply_markup)

            return self.CHECK
        else:
            message = f'({self.PRE_GAME}/{self.n}) - Select then the purpose of your trip'
            reply_markup = ReplyKeyboardMarkup(self.categories, one_time_keyboard=True, resize_keyboard=True)
            bot.message.reply_text(message, reply_markup=reply_markup)

            return self.PRP

    def create_territorial_features(self, section):
        new_obj = {}
        try:
            for t in self.territorial_features:
                new_obj[t] = self.territorial_info.loc[section, t]
        except:
            prob = ['home', 'work', 'eating', 'entertainment', 'recreation', 'shopping', 'travel', 'admin_chores',
                    'religious',
                    'health', 'police', 'education']
            order = ['work', 'home', 'travel', 'education', 'entertainment', 'recreation', 'eating', 'shopping',
                     'admin_chores', 'religious', 'police',
                     'health']
            for t in self.territorial_features:
                if t in prob:
                    lambda_ = 0.45 + np.random.uniform(low=0.01, high=0.2)
                    value = planck.pmf(order.index(t), lambda_)
                    new_obj[t] = value
                else:
                    value = self.territorial_info[t].mean()
                    new_obj[t] = value

        return new_obj

    def create_dataset(self, user):
        if self.new_user == 0:
            objs, users = self.fb.download('users')
            position_user = users.index(user)
            object_user = objs[position_user]
            self.age = object_user['age']
            self.gender = object_user['gender']
            self.occupation = object_user['occupation']

        new_obj = self.create_territorial_features(self.d_census_id)
        new_obj['activity_time'] = self.activity
        new_obj['mode'] = self.mode
        new_obj['age'] = self.age
        new_obj['gender'] = self.gender
        new_obj['occupation'] = self.occupation
        new_obj['d_hour'] = self.finish_time
        new_obj['bin_weekday'] = self.week
        x = pd.DataFrame(new_obj, index=[0], columns=self.columns)

        return x

    def check(self, bot, update):
        purpose = bot.message.text
        self.purpose = purpose
        if purpose.upper() == self.prediction:
            message = "😎"
            bot.message.reply_text(message)
        else:
            message = "😔"
            bot.message.reply_text(message)
        self.save_trip()

        return ConversationHandler.END

    def purpose(self, bot, update):
        self.purpose = bot.message.text
        logger.info(f'{self.user}-->{self.purpose}')
        message = 'Thanks for your cooperation'
        bot.message.reply_text(message)
        self.save_trip()

        return ConversationHandler.END

    def save_trip(self):
        tmp_obj = {
            'O': {
                'lat': self.o_lat,
                'lng': self.o_lng,
                'census_id': self.o_census_id,
                'name': self.origin_name
            },
            'D': {
                'lat': self.d_lat,
                'lng': self.d_lng,
                'census_id': self.d_census_id,
                'name': self.destination_name
            },
            'mode': self.mode,
            'activity_time': self.activity,
            'd_hour': self.finish_time,
            'is_week': self.week,
            'category': self.purpose,
            'user_id': self.user
        }
        print(tmp_obj)
        self.new_trips += 1
        self.fb.upload('trips', tmp_obj)
        logger.info(f'{self.user}-->SAVED ON FIREBASE')
        if self.new_trips == self.threshold_n_trips:
            self.retrain_rf()

    def retrain_rf(self):
        print('Retraining Model...')
        from ReTrain import retrain_model
        retrain_model(self.fb, self.df, self.territorial_info, self.territorial_features, self.columns,
                      th=self.threshold_n_trips)
        self.new_trips = 0

    def statistical_analysis(self, mode='diary'):
        diary, n_trips, most_category, most_destination = self.fb.specific_download("trips", self.user)
        return diary, n_trips, most_category, most_destination

    def cancel(self, bot, update):
        bot.message.reply_text('Bye! I hope we can talk again some day.',
                               reply_markup=ReplyKeyboardRemove())

        return ConversationHandler.END

    def halp(self, bot, update):
        help_message = """
			/start: use to start the bot or to re-start the bot when you want to enter new trips\n
			/cancel: use to end the conversation with the bot
		"""
        bot.update.reply_text(help_message)

        return self.START

    def main(self):
        updater = Updater(bot_key, use_context=True)
        dp = updater.dispatcher

        conv_handler = ConversationHandler(
            entry_points=[CommandHandler('start', self.start)],
            states={
                self.CHOOSE: [MessageHandler(Filters.text, self.choose)],
                self.AGE: [MessageHandler(self.age_filter, self.age)],
                self.GND: [MessageHandler(self.gender_filter, self.gender)],
                self.OCC: [MessageHandler(self.occupation_filter, self.occupation)],
                self.ORG: [MessageHandler(Filters.text, self.origin)],
                self.DEST: [MessageHandler(Filters.text, self.destination)],
                self.MODE: [MessageHandler(self.mode_filter, self.mode)],
                self.DESTINATION_START: [MessageHandler(self.destination_start_filter, self.destination_start)],
                self.ACT: [MessageHandler(self.activity_filter, self.activity)],
                self.IS_WEEK: [MessageHandler(self.is_week_filter, self.is_week)],
                self.PRE_GAME: [MessageHandler(self.game_filter, self.pre_game)],
                self.CHECK: [MessageHandler(self.category_filter, self.check)],
                self.PRP: [MessageHandler(self.category_filter, self.purpose)]
            },
            fallbacks=[CommandHandler('cancel', self.cancel), CommandHandler('help', self.halp)],
            allow_reentry=True,
            conversation_timeout=600,
        )
        dp.add_handler(conv_handler)
        updater.start_polling()
        updater.idle()


if __name__ == '__main__':
    tb = TorinoBot()
    tb.main()
